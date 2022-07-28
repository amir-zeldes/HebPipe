import conllu
import torch
import torch.nn as nn
import os
import shutil
import random
import math
import re

from flair.data import Dictionary, Sentence
from transformers import BertModel,BertTokenizerFast
from random import sample
from collections import defaultdict
from lib.crfutils.crf import CRF
from lib.crfutils.viterbi import ViterbiDecoder,ViterbiLoss
from .reorder_sgml import reorder
from .tt2conll import conllize

from time import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

class Score:
        def __init__(self, gold_total, system_total, correct, aligned_total=None):
            self.correct = correct
            self.gold_total = gold_total
            self.system_total = system_total
            self.aligned_total = aligned_total
            self.precision = correct / system_total if system_total else 0.0
            self.recall = correct / gold_total if gold_total else 0.0
            self.f1 = 2 * correct / (system_total + gold_total) if system_total + gold_total else 0.0
            self.aligned_accuracy = correct / aligned_total if aligned_total else aligned_total


class UDSpan:
    def __init__(self, start, end):
        self.start = start
        # Note that self.end marks the first position **after the end** of span,
        # so we can use characters[start:end] or range(start, end).
        self.end = end

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



class MTLModel(nn.Module):
    def __init__(self,sbdrnndim=512,posrnndim=512,sbdrnnnumlayers=2,posrnnnumlayers=2,sbdrnnbidirectional=True,posrnnbidirectional=True,sbdrnndropout=0.3,posrnndropout=0.3,sbdencodertype='lstm',posencodertype='lstm',sbdffdim=512,posffdim=512,batchsize=16,sbdtransformernumlayers=4,sbdnhead=4,sequencelength=128):
        super(MTLModel,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # tagsets - amend labels here
        self.postagset = {'ADJ':0, 'ADP':1, 'ADV':2, 'AUX':3, 'CCONJ':4, 'DET':5, 'INTJ':6, 'NOUN':7, 'NUM':8, 'PRON':9, 'PROPN':10, 'PUNCT':11, 'SCONJ':12, 'SYM':13, 'VERB':14, 'X':15} # derived from HTB and IAHLTWiki trainsets #TODO: add other UD tags?
        self.sbd_tag2idx = {'B-SENT': 1,'O': 0}

        # POS tagset in Dictionary object for Flair CRF
        self.postagsetcrf = Dictionary()
        for key in self.postagset.keys():
            self.postagsetcrf.add_item(key.strip())
        self.postagsetcrf.add_item("<START>")
        self.postagsetcrf.add_item("<STOP>")

        # shared hyper-parameters
        self.sequence_length = sequencelength
        self.batch_size = batchsize

        # Embedding parameters and model
        # Embeddings on the cpu.
        self.tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
        self.model = BertModel.from_pretrained('onlplab/alephbert-base').to(self.device)
        self.embeddingdim = 768

        # Bi-LSTM Encoder for SBD
        self.sbdrnndim = sbdrnndim
        self.sbdrnnnumlayers = sbdrnnnumlayers
        self.sbdrnnbidirectional = sbdrnnbidirectional
        self.sbdrnndropout = sbdrnndropout

        #Bi-LSTM Encoder for POS tagging
        self.posrnndim = posrnndim
        self.posrnnnumlayers = posrnnnumlayers
        self.posrnnbidirectional = posrnnbidirectional
        self.posrnndropout = posrnndropout

        if sbdencodertype == 'lstm':
            self.sbdencoder = nn.LSTM(input_size=self.embeddingdim, hidden_size=self.sbdrnndim // 2,
                                 num_layers=self.sbdrnnnumlayers, bidirectional=self.sbdrnnbidirectional,
                                 dropout=self.sbdrnndropout,batch_first=True).to(self.device)
        elif sbdencodertype == 'gru':
            self.sbdencoder = nn.GRU(input_size=self.embeddingdim, hidden_size=self.sbdrnndim // 2,
                                   num_layers=self.sbdrnnnumlayers, bidirectional=self.sbdrnnbidirectional,
                                   dropout=self.sbdrnndropout,batch_first=True).to(self.device)
        elif sbdencodertype == 'transformer':
            self.sbdtransformernumlayers = sbdtransformernumlayers
            self.sbdnhead = sbdnhead
            self.sbdencoderlayer = nn.TransformerEncoderLayer(d_model= self.embeddingdim,nhead=self.sbdnhead).to(self.device)
            self.sbdencoder = nn.TransformerEncoder(self.sbdencoderlayer,num_layers=self.sbdtransformernumlayers).to(self.device)
            self.sbdposencoder = PositionalEncoding(d_model=self.embeddingdim).to(self.device)

        # param init
        for name, param in self.sbdencoder.named_parameters():
            try:
                if 'bias' in name:
                    nn.init.constant_(param,0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
            except ValueError as ex:
                nn.init.constant_(param,0.0)

        if posencodertype == 'lstm':
            self.posencoder = nn.LSTM(input_size=self.embeddingdim + 1, hidden_size=self.posrnndim // 2,
                                 num_layers=self.posrnnnumlayers, bidirectional=self.posrnnbidirectional,
                                 dropout=self.posrnndropout,batch_first=True).to(self.device)
        elif posencodertype == 'gru':
            self.posencoder = nn.GRU(input_size=self.embeddingdim + 1, hidden_size=self.posrnndim // 2,
                                   num_layers=self.posrnnnumlayers, bidirectional=self.posrnnbidirectional,
                                   dropout=self.posrnndropout,batch_first=True).to(self.device)

        # param init
        for name, param in self.posencoder.named_parameters():
            try:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
            except ValueError as ex:
                nn.init.constant_(param, 0.0)

        self.relu = nn.ReLU()

        # Intermediate feedforward layer
        self.sbdffdim = sbdffdim
        if sbdencodertype == 'transformer':
            self.sbdfflayer = nn.Linear(in_features=self.embeddingdim, out_features=self.sbdffdim).to(self.device)
        else:
            self.sbdfflayer = nn.Linear(in_features=self.sbdrnndim, out_features=self.sbdffdim).to(self.device)

        # param init
        for name, param in self.sbdfflayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Intermediate feedforward layer
        self.posffdim = posffdim
        self.posfflayer = nn.Linear(in_features=self.posrnndim, out_features=self.posffdim).to(self.device)

        # param init
        for name, param in self.posfflayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Label space for the pos tagger
        self.hidden2postag = nn.Linear(in_features=self.posffdim,out_features=len(self.postagsetcrf)).to(self.device)
        for name, param in self.hidden2postag.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Label space for sent splitter
        self.hidden2sbd = nn.Linear(in_features=self.sbdffdim,out_features=len(self.sbd_tag2idx.keys())).to(self.device)

        # param init
        for name, param in self.hidden2sbd.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)
        self.embeddingdropout = nn.Dropout(p=0.1)

        self.poscrf = CRF(self.postagsetcrf,len(self.postagsetcrf),init_from_state_dict=False) # TODO: parameterize
        self.viterbidecoder = ViterbiDecoder(self.postagsetcrf)

        for name, param in self.poscrf.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.stride_size = 10
        self.sbdencodertype = sbdencodertype
        self.posencodertype = posencodertype

    def shingle(self,toks,labels=None):
        """
        Returns the span embeddings, labelspans, and 'final mapping'
        """
        spans = []
        labelspans = []
        final_mapping = {}

        # Hack tokens up into overlapping shingles
        wraparound = torch.cat((toks[-self.stride_size:],toks,toks[: self.sequence_length]),dim=0)
        if labels:
            labelwraparound = labels[-self.stride_size:] + labels + labels[: self.sequence_length]
        idx = 0
        mapping = defaultdict(set)
        snum = 0
        while idx < len(toks):
            if idx + self.sequence_length < len(wraparound):
                span = wraparound[idx: idx + self.sequence_length]
                if labels:
                    labelspan = labelwraparound[idx: idx + self.sequence_length]
            else:
                span = wraparound[idx:]
                if labels:
                    labelspan = labelwraparound[idx:]

            spans.append(span)
            if labels:
                labelspans.append(labelspan)

            for i in range(idx - self.stride_size, idx + self.sequence_length - self.stride_size):
                # start, end, snum
                if i >= 0 and i < len(toks):
                    mapping[i].add(
                        (idx - self.stride_size, idx + self.sequence_length - self.stride_size, snum))
            idx += self.stride_size
            snum += 1

        for idx in mapping:
            best = self.sequence_length
            for m in mapping[idx]:
                start, end, snum = m
                dist_to_end = end - idx
                dist_to_start = idx - start
                delta = abs(dist_to_end - dist_to_start)
                if delta < best:
                    best = delta
                    final_mapping[idx] = (snum, idx - start)  # Get sentence number and position in sentence

        spans = torch.stack(spans)

        return spans,labelspans,final_mapping

    def forward(self,data,mode='train'):

        badrecords = [] # stores records where AlephBERT's tokenization 'messed up' the sentence's sequence length, and removes these sentences from the batch.

        # Extract the sentences and labels
        if mode == 'train': # training is on a batch, so 3D tensor
            sentences = [' '.join([s.split('\t')[0].strip() for s in sls]) for sls in data]
            sbdlabels = [[self.sbd_tag2idx[s.split('\t')[2].strip()] for s in sls] for sls in data]
            poslabels = [[self.postagsetcrf.get_idx_for_item(s.split('\t')[1].strip()) for s in sls] for sls in data]
        elif mode == 'dev': # inference is on a single record
            sentences = [' '.join([s.split('\t')[0].strip() for s in data])]
            sbdlabels = [self.sbd_tag2idx[s.split('\t')[2].strip()] for s in data]
            poslabels = [self.postagsetcrf.get_idx_for_item(s.split('\t')[1].strip()) for s in data]
        else: # test - has no labels, and 2D tensor single record
            sentences = [' '.join([s.split('\t')[0].strip() for s in data])]
            sbdlabels = None
            poslabels = None

        # Make embeddings
        sentences = [d.split() for d in sentences] # for AlephBERT
        tokens = self.tokenizer(sentences,return_tensors='pt',padding=True,is_split_into_words=True).to(self.device) # tell AlephBERT that there is some tokenization already.
        embeddings = self.model(**tokens)
        embeddings = embeddings[0]
        #embeddings = embeddings.to(self.device)

        """
        Average the subword embeddings
        This process will drop the [CLS],[SEP] and [PAD] tokens
        """

        avgembeddings = []
        for k in range(0,len(tokens.encodings)):
            emb = []
            maxindex = max([w for w in tokens.encodings[k].words if w is not None])

            try:
                assert maxindex == self.sequence_length - 1  # otherwise won't average correctly and align with labels
            except AssertionError:
                #print ('max index not equal sequence len. Default labels will be applied.')
                badrecords.append(k)
                continue

            for i in range(0,self.sequence_length):

                indices = [j for j,x in enumerate(tokens.encodings[k].words) if x == i]
                if len(indices) == 0: # This strange case needs to be handled.
                    emb.append(torch.zeros(self.embeddingdim,device=self.device))
                elif len(indices) == 1: # no need to average
                    emb.append(embeddings[k][indices[0]])
                else: # needs to aggregate - average
                    slice = embeddings[k][indices[0]:indices[-1] + 1]
                    slice = torch.mean(input=slice,dim=0,keepdim=False)
                    emb.append(slice)
            try:
                assert len(emb) == self.sequence_length # averaging was correct and aligns with the labels
            except AssertionError:
                #print ('embedding not built correctly. Default labels will be applied')
                badrecords.append(k)
                continue

            emb = torch.stack(emb)
            avgembeddings.append(emb)

        badrecords = sorted(badrecords,reverse=True)

        avgembeddings = torch.stack(avgembeddings)
        for record in badrecords:
            sbdlabels.pop(record)
            poslabels.pop(record)
            self.batch_size -= 1

        if mode != 'train':
            if avgembeddings.size(dim=1) > self.stride_size: # don't shingle if seqlen less than the overlap
                # squeeze the embedding, as it's a single sentence
                avgembeddings = torch.squeeze(avgembeddings)
                # shingle the sentence embedding and its label, to calculate the dev loss later
                sbdembeddings,finalsbdlabels,finalmapping = self.shingle(avgembeddings,sbdlabels)
                # restore dimensionality for the POS tagging pipeline.
                avgembeddings = torch.unsqueeze(avgembeddings,dim=0)
            else:
                sbdembeddings = avgembeddings
                finalsbdlabels = sbdlabels
                finalmapping = None
        else:
            sbdembeddings = avgembeddings
            finalsbdlabels = sbdlabels
            finalmapping = None

        sbdembeddings = self.embeddingdropout(sbdembeddings)

        # SBD encoder and labels
        if self.sbdencodertype in ('lstm','gru'):
            feats, _ = self.sbdencoder(sbdembeddings)
        else:
            feats = self.sbdposencoder(sbdembeddings)
            feats = self.sbdencoder(feats)

        # SBD Intermediate Feedforward layer
        feats = self.sbdfflayer(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)

        # SBD logits
        sbdlogits = self.hidden2sbd(feats)

        #get the sbd predictions as input to the POS encoder
        if mode == 'train':
            sbdpreds = torch.argmax(sbdlogits,dim=2,keepdim=True)
        else:
            # Predict from the shingles for SBD.
            # 'Believe the span where the token is most in the middle'
            if sbdlogits.size(dim=1) > self.stride_size:
                sbdpreds = []
                for idx in finalmapping:
                    snum, position = finalmapping[idx]
                    label = torch.argmax(sbdlogits[snum][position]).item()
                    sbdpreds.append(label)

                # Unsqueeze for input to the POS Encoder
                sbdpreds = torch.LongTensor(sbdpreds)
                sbdpreds = torch.unsqueeze(sbdpreds, dim=0)
                sbdpreds = torch.unsqueeze(sbdpreds, dim=2)
                sbdpreds = sbdpreds.to(self.device)
            else:
                sbdpreds = torch.argmax(sbdlogits, dim=2, keepdim=True)

        # Add the SBD predictions to the POS Encoder Input!
        posembeddings = torch.cat((avgembeddings,sbdpreds),dim=2)

        if mode in ('dev','test'):
            # Squeeze these to return to the Trainer for scores, now that we are done with them
            sbdpreds = torch.squeeze(sbdpreds,dim=2)
            sbdpreds = torch.squeeze(sbdpreds, dim=0)
            sbdpreds = sbdpreds.tolist()
        else:
            sbdpreds = None

        if self.posencodertype in ('lstm','gru'):
            feats,_ = self.posencoder(posembeddings)

        # logits for pos
        feats = self.posfflayer(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        poslogits = self.hidden2postag(feats)
        poslogits = self.poscrf(poslogits)

        return sbdlogits, finalsbdlabels, sbdpreds, poslogits, poslabels # returns the logits and labels

class Tagger():
    def __init__(self,trainflag=False,trainfile=None,devfile=None,testfile=None,sbdrnndim=512,sbdrnnnumlayers=2,sbdrnnbidirectional=True,sbdrnndropout=0.3,sbdencodertype='lstm',sbdffdim=512,learningrate = 0.001,bestmodelpath='../data/checkpoint/'):

        self.mtlmodel = MTLModel(sbdrnndim=sbdrnndim,sbdrnnnumlayers=sbdrnnnumlayers,sbdrnnbidirectional=sbdrnnbidirectional,sbdrnndropout=sbdrnndropout,sbdencodertype=sbdencodertype,sbdffdim=sbdffdim)

        if trainflag == True:

            from torch.utils.tensorboard import SummaryWriter
            if os.path.isdir('../data/tensorboarddir/'):
                shutil.rmtree('../data/tensorboarddir/')
            os.mkdir('../data/tensorboarddir/')

            if not os.path.isdir(bestmodelpath):
                os.mkdir(bestmodelpath)

            self.writer = SummaryWriter('../data/tensorboarddir/')

            self.trainingdatafile = '../data/sentsplit_postag_train_gold.tab'
            self.devdatafile = '../data/sentsplit_postag_dev_gold.tab'

            self.bestmodel = bestmodelpath + 'best_sent_pos_model.pt'

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.trainflag = trainflag
        self.trainfile = trainfile
        self.devfile = devfile
        self.testfile = testfile

        self.learningrate = learningrate

        # Loss for pos tagging
        self.postagloss = ViterbiLoss(self.mtlmodel.postagsetcrf)
        self.postagloss.to(self.device)

        # Loss for sentence splitting
        self.sbdloss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,3]))
        self.sbdloss.to(self.device)

        self.optimizer = torch.optim.AdamW(list(self.mtlmodel.sbdencoder.parameters()) +  list(self.mtlmodel.sbdfflayer.parameters()) +
                                           list(self.mtlmodel.hidden2sbd.parameters()) + list(self.mtlmodel.posencoder.parameters()) + list(self.mtlmodel.posfflayer.parameters())
                                           + list(self.mtlmodel.hidden2postag.parameters()) + list(self.mtlmodel.poscrf.parameters()), lr=learningrate)

        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[400,1000],gamma=0.1)
        self.evalstep = 20

    def set_seed(self, seed):

        random.seed(seed)
        torch.manual_seed(seed)

    def train(self,checkpointfile=None):

        def read_file(mode='train'):

            dataset = []
            if mode == 'dev':
                with open(self.devdatafile, 'r') as fi:
                    lines = fi.readlines()
                    #lines = list(reversed(lines))  # hebrew is right to left...

                    for idx in range(0, len(lines), self.mtlmodel.sequence_length):
                        if idx + self.mtlmodel.sequence_length >= len(lines):
                            slice = lines[idx:len(lines)]
                        else:
                            slice = lines[idx: idx + self.mtlmodel.sequence_length]

                        dataset.append(slice)

                test = [d for slice in dataset for d in slice]
                assert len(test) == len(lines)

            else:
                with open(self.trainingdatafile,'r') as fi:
                    lines = fi.readlines()
                    #lines = list(reversed(lines)) # hebrew is right to left...

                    # shingle it here to get more training data
                    for idx in range(0,len(lines),self.mtlmodel.sequence_length - self.mtlmodel.stride_size):
                        if idx + self.mtlmodel.sequence_length >= len(lines):
                            slice = lines[idx:len(lines)]
                            dataset.append(slice)
                            break
                        else:
                            slice = lines[idx: idx + self.mtlmodel.sequence_length]
                            dataset.append(slice)

            return dataset

        epochs = 1000
        bestloss = float('inf')

        trainingdata = read_file()
        devdata = read_file(mode='dev')

        if checkpointfile is not None:
            checkpoint = torch.load(checkpointfile)
            self.mtlmodel.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.mtlmodel.poscrf.load_state_dict(checkpoint['poscrf_state_dict'])

        self.set_seed(42)

        for epoch in range(1,epochs):

            old_batchsize = self.mtlmodel.batch_size

            self.mtlmodel.train()
            self.optimizer.zero_grad()

            data = sample(trainingdata,self.mtlmodel.batch_size)
            data = [datum for datum in data if len(datum) == self.mtlmodel.sequence_length]
            self.mtlmodel.batch_size = len(data)

            sbdlogits, sbdlabels, _, poslogits,poslabels = self.mtlmodel(data)

            sbdtags = torch.LongTensor(sbdlabels).to(self.device)
            sbdlogits = sbdlogits.permute(0,2,1)
            sbdloss = self.sbdloss(sbdlogits,sbdtags)

            lengths = [self.mtlmodel.sequence_length] * self.mtlmodel.batch_size
            lengths = torch.LongTensor(lengths).to(self.device)
            scores = (poslogits, lengths, self.mtlmodel.poscrf.transitions)

            # unwrap the pos tags into one long list first
            postags = [p for pos in poslabels for p in pos]
            postags = torch.LongTensor(postags).to(self.device)
            posloss = self.postagloss(scores,postags)


            mtlloss = posloss + sbdloss # TODO: learnable weights?
            mtlloss.backward()
            self.optimizer.step()
            self.scheduler.step() # TODO: Multi-step LR annealing seems to increase sentence splitting performance. Need a best annealing strategy

            if old_batchsize != self.mtlmodel.batch_size:
                self.mtlmodel.batch_size = old_batchsize

            self.writer.add_scalar('train_pos_loss', posloss.item(), epoch)
            self.writer.add_scalar('train_sbd_loss', sbdloss.item(), epoch)
            self.writer.add_scalar('train_joint_loss', mtlloss.item(), epoch)

            """""""""""""""""""""""""""""""""""""""""""""
            Do dev evaluation after evalstep number of epochs
            """""""""""""""""""""""""""""""""""""""""""""
            if epoch % self.evalstep == 0:

                self.mtlmodel.eval()

                with torch.no_grad():

                    totalsbddevloss = 0
                    totalposdevloss = 0

                    allsbdpreds = []
                    allsbdgold = []
                    allpospreds = []
                    allposgold = []

                    start = time()

                    # because of shingling for SBD, the dev data needs to be split in slices for inference, as GPU may run out of memory with shingles on the full token list.
                    # shingling and SBD prediction is done on the individual slice, as well as POS tag predictions.
                    # TODO This naturally increases prediction time...but can't think of a better way.
                    for slice in devdata:

                        old_seqlen = self.mtlmodel.sequence_length
                        if len(slice) != self.mtlmodel.sequence_length: # this will happen in one case, for the last slice in the dev batch
                            self.mtlmodel.sequence_length = len(slice)

                        # Flair CRF decoding uses the Sentence object..
                        sentence = ' '.join([s.split('\t')[0].strip() for s in slice])
                        sentence = Sentence(sentence,use_tokenizer=False)

                        goldsbdlabels = [s.split('\t')[2].strip() for s in slice]
                        goldsbdlabels = [self.mtlmodel.sbd_tag2idx[s] for s in goldsbdlabels]
                        goldposlabels = [s.split('\t')[1].strip() for s in slice]
                        goldposlabels = [self.mtlmodel.postagsetcrf.get_idx_for_item(s) for s in goldposlabels]

                        # sbdpreds already contains the sbd predictions. These were necessary for input to the POS encoder.
                        sbdlogits, sbdlabels, sbdpreds, poslogits, poslabels = self.mtlmodel(slice,mode='dev')

                        # get the pos predictions
                        lengths = [self.mtlmodel.sequence_length]
                        lengths = torch.LongTensor(lengths).to(self.device)
                        scores = (poslogits, lengths, self.mtlmodel.poscrf.transitions)
                        pospreds = self.mtlmodel.viterbidecoder.decode(scores,False,[sentence])
                        pospreds = [self.mtlmodel.postagsetcrf.get_idx_for_item(p[0]) for pr in pospreds[0] for p in pr]

                        # get the sbd loss
                        sbdlogits = sbdlogits.permute(0,2,1)
                        sbdtags = torch.LongTensor(sbdlabels).to(self.device)
                        sbddevloss = self.sbdloss(sbdlogits, sbdtags).item()

                        # get the pos loss
                        postags = torch.LongTensor(poslabels)
                        postags = postags.to(self.device)
                        posdevloss = self.postagloss(scores,postags).item()

                        totalsbddevloss += sbddevloss
                        totalposdevloss += posdevloss

                        allsbdpreds.extend(sbdpreds)
                        allsbdgold.extend(goldsbdlabels)
                        allpospreds.extend(pospreds)
                        allposgold.extend(goldposlabels)

                    #print ('inference time')
                    #print (time() - start)
                    if self.mtlmodel.sequence_length != old_seqlen:
                        self.mtlmodel.sequence_length = old_seqlen

                    goldspans = []
                    predspans = []
                    goldstartindex = 0
                    predstartindex = 0

                    for i in range(0,len(allsbdgold)):
                        if allsbdgold[i] == 1: #B-SENT
                            goldspans.append(UDSpan(goldstartindex,i))
                            goldstartindex = i
                        if allsbdpreds[i] == 1:
                            predspans.append(UDSpan(predstartindex,i))
                            predstartindex = i

                    sbdscores = self.spans_score(goldspans,predspans)

                    correctpos = sum([1 if p == g else 0 for p,g in zip(allpospreds,allposgold)])
                    posscores = Score(len(allposgold),len(allpospreds),correctpos,len(allpospreds))

                    mtlloss = (totalsbddevloss + totalposdevloss) / len(devdata)

                    self.writer.add_scalar("mtl_dev_loss", round(mtlloss, 4),
                                           int(epoch / self.evalstep))
                    print('mtl dev loss:' + str(round((totalsbddevloss / len(devdata) + (totalposdevloss / len(devdata))), 4)))

                    self.writer.add_scalar("sbd_dev_loss",round(totalsbddevloss/len(devdata),4),int(epoch / self.evalstep))
                    self.writer.add_scalar("sbd_dev_f1", round(sbdscores.f1,4), int(epoch / self.evalstep))
                    self.writer.add_scalar("sbd_dev_precision", round(sbdscores.precision, 4), int(epoch / self.evalstep))
                    self.writer.add_scalar("sbd_dev_recall", round(sbdscores.recall, 4), int(epoch / self.evalstep))

                    print ('\n')
                    self.writer.add_scalar("pos_dev_loss", round(totalposdevloss / len(devdata), 4),
                                           int(epoch / self.evalstep))
                    self.writer.add_scalar("pos_dev_f1", round(posscores.f1, 4), int(epoch / self.evalstep))
                    self.writer.add_scalar("pos_dev_precision", round(posscores.precision, 4),
                                           int(epoch / self.evalstep))
                    self.writer.add_scalar("pos_dev_recall", round(posscores.recall, 4), int(epoch / self.evalstep))

                    print ('sbd dev f1:' + str(sbdscores.f1))
                    print('sbd dev precision:' + str(sbdscores.precision))
                    print('sbd dev recall:' + str(sbdscores.recall))
                    print ('\n')

                    print('pos dev f1:' + str(posscores.f1))
                    print('pos dev precision:' + str(posscores.precision))
                    print('pos dev recall:' + str(posscores.recall))
                    print('\n')

                    if mtlloss < bestloss:
                        bestloss = mtlloss
                        bestmodel = self.bestmodel.replace('.pt','_' + str(round(mtlloss,6)) + '_' + str(round(sbdscores.f1,6)) + '_' + str(round(posscores.f1,6)) + '.pt')
                        torch.save({'epoch':epoch,'model_state_dict':self.mtlmodel.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),'poscrf_state_dict':self.mtlmodel.poscrf.state_dict()},bestmodel)

    def predict(self,toks,checkpointfile=None):


        def is_tok(sgml_line):
            return len(sgml_line) > 0 and not (sgml_line.startswith("<") and sgml_line.endswith(">"))

        def unescape(token):
            token = token.replace("&quot;", '"')
            token = token.replace("&lt;", "<")
            token = token.replace("&gt;", ">")
            token = token.replace("&amp;", "&")
            token = token.replace("&apos;", "'")
            return token

        slices = []
        toks = unescape(toks)  # Splitter is trained on UTF-8 forms, since LM embeddings know characters like '&'
        lines = toks.strip().split("\n")
        toks = [l for l in lines if is_tok(l)]
        toks = [re.sub(r"\t.*", "", t) for t in toks]

        # slice up the token list into slices of seqlen for GPU RAM reasons
        for idx in range(0, len(toks), self.mtlmodel.sequence_length):
            if idx + self.mtlmodel.sequence_length >= len(toks):
                slice = toks[idx:len(toks)]
            else:
                slice = toks[idx: idx + self.mtlmodel.sequence_length]

            slices.append(slice)

        test = [d for slice in slices for d in slice]

        assert len(test) == len(toks)



        if checkpointfile is not None:

            checkpoint = torch.load(checkpointfile)
            self.mtlmodel.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.mtlmodel.poscrf.load_state_dict(checkpoint['poscrf_state_dict'])

        self.mtlmodel.eval()

        with torch.no_grad():

            allsbdpreds = []
            allpospreds = []

            for slice in slices:

                if len(slice) != self.mtlmodel.sequence_length:  # this will happen in one case, for the last slice in the batch
                    self.mtlmodel.sequence_length = len(slice)

                # Flair CRF decoding uses the Sentence object..
                sentence = ' '.join([s.split('\t')[0].strip() for s in slice])
                sentence = Sentence(sentence, use_tokenizer=False)

                _, _, sbdpreds, poslogits, _ = self.mtlmodel(slice, mode='test')

                # get the pos predictions
                lengths = [self.mtlmodel.sequence_length]
                lengths = torch.LongTensor(lengths).to(self.device)
                scores = (poslogits, lengths, self.mtlmodel.poscrf.transitions)
                pospreds = self.mtlmodel.viterbidecoder.decode(scores, False, [sentence])
                pospreds = [self.mtlmodel.postagsetcrf.get_idx_for_item(p[0]) for pr in pospreds[0] for p in pr]

                allsbdpreds.extend(sbdpreds)
                allpospreds.extend(pospreds)

        allpospreds = [self.mtlmodel.postagsetcrf.get_item_for_index(p) for p in allpospreds]

        return allsbdpreds,allpospreds

    def prepare_data_files(self):
        """
        Prepares the train and dev data files for training
        """
        def write_file(filename,mode='train'):

            if mode == 'dev':
                data = devdata
            else:
                data = traindata

            with open(filename,'w') as tr:
                for sent in data:
                    for i in range(0,len(sent)):
                        if isinstance(sent[i]['id'], tuple): continue # MWE conventions in the conllu file

                        if sent[i]['id'] == 1:
                            tr.write(sent[i]['form'] + '\t' + sent[i]['upos'] + '\t' + 'B-SENT' + '\n')

                        else:
                            tr.write(sent[i]['form'] + '\t' + sent[i]['upos'] + '\t' + 'O' + '\n')

        traindata = self.read_conllu()
        devdata = self.read_conllu(mode='dev')

        write_file(self.trainingdatafile,mode='train')
        write_file(self.devdatafile,mode='dev')

    def read_conllu(self,mode='train'):

        fields = tuple(
            list(conllu.parser.DEFAULT_FIELDS)
        )

        if mode == 'dev':
            file = self.devfile
        else:
            file = self.trainfile

        with open(file, "r", encoding="utf-8") as f:
            return conllu.parse(f.read(), fields=fields)

    def split_pos(self, xml_data,out_mode='conllu',checkpointfile = None):

        def is_sgml_tag(line):
            return line.startswith("<") and line.endswith(">")

        def wrap_words(xml):
            output = []
            lines = xml.split("\n")
            for line in lines:
                if len(line)>0 and not (line.startswith("<") and line.endswith(">") and not line == "|"):
                    line = line.replace("|","\n")
                    line = "<❦♥>\n" + line + "\n</❦♥>"
                output.append(line)
            return "\n".join(output)

        def collapse_words(sgml):
            output = []
            buffer = []
            for line in sgml.split("\n"):
                if line in ['<❦♥>','</❦♥>'] or not is_sgml_tag(line):
                    buffer.append(line)
                else:
                    output.append(line)
                if line == "</❦♥>":
                    piped = "|".join(buffer)
                    if not (buffer[1] == "|" and len(buffer) == 3):  # Actual pipe as token
                        piped = piped.replace('|</❦♥>','</❦♥>').replace('<❦♥>|','<❦♥>')
                    output.append(piped)
                    buffer = []
            return "\n".join(output)

        def fix_malformed_sentences(sgml_list):
            """
            Fixing malformed SGML seems to boil down to two cases:

            (1) The sentence is interrupted by the close of a tag that opened before it. In this case,
                update the s boundaries so that we close and begin sentences at the close tag:

                                     <a>
                        <a>          ...
                        ...          <s>
                        <s>          ...
                        ...    ==>   </s>
                        </a>         </a>
                        ...          <s>
                        </s>         ...
                                     </s>

            (2) Some tag opened inside of the sentence and has remained unclosed at the time of sentence closure.
                In this case, we choose not to believe the sentence split, and merge the two sentences:

                        <s>
                        ...          <s>
                        <a>          ...
                        ...          <a>
                        </s>   ==>   ...
                        <s>          ...
                        ...          </a>
                        </a>         ...
                        ...          </s>
                        </s>
            """
            tag_opened = defaultdict(list)
            i = 0
            while i < len(sgml_list):
                line = sgml_list[i].strip()
                open_match = re.search(OPEN_SGML_ELT, line)
                close_match = re.search(CLOSE_SGML_ELT, line)
                if open_match:
                    tag_opened[open_match.groups()[0]].append(i)
                elif close_match:
                    tagname = close_match.groups()[0]
                    j = maximal_nontoken_span_end(sgml_list, i + 1)
                    mns = sgml_list[i:j]

                    # case 1: we've encountered a non-s closing tag. If...
                    if (
                            tagname != "s"  # the closing tag is not an s
                            and len(tag_opened["s"]) > 0  # and we're in a sentence
                            and len(tag_opened[tagname]) > 0
                            and len(tag_opened["s"]) > 0  # and the sentence opened after the tag
                            and tag_opened[tagname][-1] < tag_opened["s"][-1]
                            and "</s>" not in mns  # the sentence is not closed in the mns
                    ):
                        # end sentence here and move i back to the line we were looking at
                        sgml_list.insert(i, "</s>")
                        i += 1
                        # open a new sentence at the end of the mns and note that we are no longer in the sentence
                        sgml_list.insert(j + 1, "<s>")
                        tag_opened["s"].pop(-1)
                        # we have successfully closed this tag
                        tag_opened[tagname].pop(-1)
                    # case 2: s closing tag and there's some tag that opened inside of it that isn't closed in time
                    elif tagname == "s" and any(
                            e != "s" and f"</{e}>" not in mns
                            for e in [
                                e
                                for e in tag_opened.keys()
                                if len(tag_opened[e]) > 0 and len(tag_opened["s"]) > 0 and tag_opened[e][-1] >
                                   tag_opened["s"][-1]
                            ]
                    ):
                        # some non-s element opened within this sentence and has not been closed even in the mns
                        assert "<s>" in mns
                        sgml_list.pop(i)
                        i -= 1
                        sgml_list.pop(i + mns.index("<s>"))
                    else:
                        tag_opened[tagname].pop(-1)
                i += 1
            return sgml_list

        def maximal_nontoken_span_end(sgml_list, i):
            """Return j such that sgml_list[i:j] does not contain tokens
            and no element that is begun in the MNS is closed in it."""
            opened = []
            j = i
            while j < len(sgml_list):
                line = sgml_list[j]
                open_match = re.match(OPEN_SGML_ELT, line)
                close_match = re.match(CLOSE_SGML_ELT, line)
                if not (open_match or close_match):
                    break
                if open_match:
                    opened.append(open_match.groups()[0])
                if close_match and close_match.groups()[0] in opened:
                    break
                j += 1
            return j

        def get_bound_group_map(data):

            mapping = {}
            data = data.split("\n")
            # Ignore markup
            data = [u for u in data if not (u.startswith("<") and u.endswith(">"))]
            counter = 0
            for i, line in enumerate(data):
                super_token = line.replace("|", "") if line != "|" else "|"
                segs = line.split("|") if line != "|" else ["|"]
                for j, seg in enumerate(segs):
                    if len(segs) > 1 and j == 0:
                        mapping[counter] = (super_token, len(segs))
                        super_token = ""
                    counter += 1

            return mapping

        # These XML tags force a sentence break in the data, you can add more here:
        BLOCK_TAGS = ["sp", "head", "p", "figure", "caption", "list", "item"]
        BLOCK_TAGS += ["❦❦❦"]  # reserved tag for sentences in input based on newlines
        OPEN_SGML_ELT = re.compile(r"^<([^/ ]+)( .*)?>$")
        CLOSE_SGML_ELT = re.compile(r"^</([^/]+)>$")

        # Sometimes the tokenizer doesn't newline every elt
        xml_data = xml_data.replace("><", ">\n<")
        # Ad hoc fix for a tokenization error
        xml_data = xml_data.replace("°<", "°\n<")
        # Remove empty elements?
        # for elt in TAGS:
        #    xml_data = xml_data.replace(f"<{elt}>\n</{elt}>\n", "")
        xml_data = wrap_words(xml_data)

        # don't feed the sentencer our pos and lemma predictions, if we have them
        no_pos_lemma = re.sub(r"([^\n\t]*?)\t[^\n\t]*?\t[^\n\t]*?\n", r"\1\n", xml_data)
        split_indices, pos_tags = self.predict(no_pos_lemma,checkpointfile=checkpointfile)

        # for xml
        counter = 0
        splitted = []
        opened_sent = False
        para = True

        xml_data = xml_data.replace("<s>","<❦❦❦>").replace("</s>","</❦❦❦>")
        for line in xml_data.strip().split("\n"):
            if not is_sgml_tag(line):
                # Token
                if split_indices[counter] == 1 or para:
                    if opened_sent:
                        rev_counter = len(splitted) - 1
                        while is_sgml_tag(splitted[rev_counter]) and rev_counter > 0:
                            rev_counter -= 1
                        if rev_counter > 0:
                            splitted.insert(rev_counter + 1, "</s>")
                    splitted.append("<s>")
                    opened_sent = True
                    para = False
                counter += 1
            elif any(f"<{elt}>" in line for elt in BLOCK_TAGS) or any(
                f"</{elt}>" in line for elt in BLOCK_TAGS
            ):  # New block, force sentence split
                para = True
            splitted.append(line)

        if opened_sent:
            rev_counter = len(splitted) - 1
            while is_sgml_tag(splitted[rev_counter]):
                rev_counter -= 1
            splitted.insert(rev_counter + 1, "</s>")

        lines = "\n".join(splitted)
        lines = re.sub(r'</?❦❦❦>\n?','',lines)
        lines = reorder(lines, priorities=["s","❦♥"])
        lines = collapse_words(lines)

        # destroy any xml inside supertokens
        while re.search(r'(<❦♥>[^<>]*)<[^❦♥]+>',lines) is not None:
            lines = re.sub(r'(<❦♥>[^<>]*)<[^❦♥]+>([^<>]*</❦♥>)',r'\1\2',lines)

        # remove word and sent wrappers
        lines = re.sub(r'</?❦♥>','',lines)

        lines = reorder(lines)
        lines = fix_malformed_sentences(lines.split("\n"))
        lines = "\n".join(lines)
        lines = reorder(lines)

        # Split out the internal tags within MWT tokens, as these too get a POS tag
        lines = lines.split("\n")
        retokenized = []
        for line in lines:
            if line == "|":
                retokenized.append(line)
            else:
                retokenized.append("\n".join(line.split("|")))
        lines = "\n".join(retokenized)

        """
        Now add the pos tags
        """
        bound_group_map = get_bound_group_map(lines) if out_mode == "conllu" else None
        data = conllize(lines, element="s", super_mapping=bound_group_map, attrs_as_comments=True)
        data = data.strip() + "\n"  # Ensure final new line for last sentence

        # add the pos tags to conllized file and remove the rows hyphenated MWT ID
        output = []
        tid = 1
        k = 0
        data = data.split('\n')
        for i in range(0,len(data)):
            if len(data[i].strip())==0:
                output.append("")
                tid = 1
            else:
                if "\t" in data[i]:
                    fields = data[i].split("\t")
                    if "." in fields[0] or "-" in fields[0]: continue
                    else:
                        fields = [str(tid), fields[1].strip(), "_", pos_tags[k].strip(), pos_tags[k].strip(), "_", "_", "_", "_", "_"]
                        output.append('\t'.join(fields))
                        tid += 1
                        k += 1

        assert k == len(pos_tags) # Fails means pos tags aren't aligned with tokens

        return "\n".join(output), lines

    def spans_score(self, gold_spans, system_spans):
        correct, gi, si = 0, 0, 0
        while gi < len(gold_spans) and si < len(system_spans):
            if system_spans[si].start < gold_spans[gi].start:
                si += 1
            elif gold_spans[gi].start < system_spans[si].start:
                gi += 1
            else:
                correct += gold_spans[gi].end == system_spans[si].end
                si += 1
                gi += 1

        return Score(len(gold_spans), len(system_spans), correct)



def main(): # testing only

    iahltwikitrain = '/home/nitin/Desktop/IAHLT/UD_Hebrew-IAHLTwiki/he_iahltwiki-ud-train.conllu'
    iahltwikidev = '/home/nitin/Desktop/IAHLT/UD_Hebrew-IAHLTwiki/he_iahltwiki-ud-dev.conllu'

    tagger = Tagger(trainflag=True,trainfile=iahltwikitrain,devfile=iahltwikidev)
    tagger.prepare_data_files()
    tagger.train()



if __name__ == "__main__":
    main()
