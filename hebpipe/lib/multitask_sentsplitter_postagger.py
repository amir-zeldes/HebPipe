import conllu
import torch
import torch.nn as nn
import os
import shutil
import random
import math

from flair.data import Sentence, Dictionary
from transformers import BertModel,BertTokenizerFast
from random import sample
from collections import defaultdict
from lib.crfutils.crf import CRF
from lib.crfutils.viterbi import ViterbiDecoder,ViterbiLoss

from time import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"
SAMPLE_SIZE = 16


def spans_score(gold_spans, system_spans):
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
    def __init__(self,sbdrnndim=512,posrnndim=512,sbdrnnnumlayers=2,posrnnnumlayers=2,sbdrnnbidirectional=True,posrnnbidirectional=True,sbdrnndropout=0.3,posrnndropout=0.3,sbdencodertype='lstm',posencodertype='lstm',sbdffdim=512,posffdim=512,batchsize=SAMPLE_SIZE,sbdtransformernumlayers=6,sbdnhead=8,sequencelength=128):
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
            self.posencoder = nn.LSTM(input_size=self.embeddingdim, hidden_size=self.posrnndim // 2,
                                 num_layers=self.posrnnnumlayers, bidirectional=self.posrnnbidirectional,
                                 dropout=self.posrnndropout,batch_first=True).to(self.device)
        elif posencodertype == 'gru':
            self.posencoder = nn.GRU(input_size=self.embeddingdim, hidden_size=self.posrnndim // 2,
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
        self.hidden2postag = nn.Linear(in_features=self.posffdim,out_features=len(self.postagset.keys())).to(self.device)
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

        self.stride_size = 10
        self.sbdencodertype = sbdencodertype

    def shingle(self,toks,labels=None):
        """
        Returns the span embeddings, labelspans, and 'final mapping'
        """
        spans = []
        labelspans = []
        final_mapping = {}

        # Hack tokens up into overlapping shingles
        #wraparound = toks[-self.stride_size:] + toks + toks[: self.mtlmodel.sequence_length]
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

        badrecords = [] # stores records where AlephBERT's tokenization messed up the sentence's sequence length, and removes these sentences from the batch.

        if mode == 'train': # training is on a batch, so 3D tensor
            sentences = [' '.join([s.split('\t')[0].strip() for s in sls]) for sls in data]
            sbdlabels = [[self.sbd_tag2idx[s.split('\t')[2].strip()] for s in sls] for sls in data]
        elif mode == 'dev': # inference is on a single record, 2D tensor
            sentences = [' '.join([s.split('\t')[0].strip() for s in data])]
            sbdlabels = [s.split('\t')[2].strip() for s in data]
        else: # test - has no labels, and 2D tensor single record
            sentences = [s.split('\t')[0].strip() for s in data]
            sbdlabels = None

        sentences = [d.split() for d in sentences] # for AlephBERT
        tokens = self.tokenizer(sentences,return_tensors='pt',padding=True,is_split_into_words=True).to(self.device) # tell AlephBERT that there is some tokenization already. Otherwise its own subword tokenization messes things up.

        embeddings = self.model(**tokens)
        embeddings = embeddings[0]

        """
        Average the subword embeddings
        This process will drop the [CLS],[SEP] and [PAD] tokens
        """
        #start = time()
        avgembeddings = []
        for k in range(0,len(tokens.encodings)):
            emb = []
            maxindex = max([w for w in tokens.encodings[k].words if w])

            try:
                assert maxindex == self.sequence_length - 1  # otherwise won't average correctly and align with labels
            except AssertionError:
                print ('max index not equal sequence len. Skipping.')
                badrecords.append(k)
                continue

            for i in range(0,self.sequence_length):

                indices = [j for j,x in enumerate(tokens.encodings[k].words) if x == i]
                if len(indices) == 0: # This strange case needs to be handled.
                    emb.append(torch.zeros(768,device=self.device))
                elif len(indices) == 1: # no need to average
                    emb.append(embeddings[k][indices[0]])
                else: # needs to aggregate - average
                    slice = embeddings[k][indices[0]:indices[-1] + 1]
                    slice = torch.mean(input=slice,dim=0,keepdim=False)
                    emb.append(slice)
            try:
                assert len(emb) == self.sequence_length # averaging was correct and aligns with the labels
            except AssertionError:
                print ('embedding not built correctly. Skipping')
                badrecords.append(k)
                continue

            emb = torch.stack(emb)
            avgembeddings.append(emb)

        badrecords = sorted(badrecords,reverse=True)

        if len(avgembeddings) > 0:
            avgembeddings = torch.stack(avgembeddings)
            for record in badrecords:
                sbdlabels.pop(record)
        else:
            return None,None,None

        #print ('average embeddings')
        #print (time() - start)

        if mode != 'train':
            # squeeze the embedding, as it's a single sentence
            avgembeddings = torch.squeeze(avgembeddings)
            finalembeddings,finallabels,finalmapping = self.shingle(avgembeddings,sbdlabels)
            if mode != 'test':
                finallabels = [[self.sbd_tag2idx[s] for s in sls] for sls in finallabels]
        else:
            finalembeddings = avgembeddings
            finallabels = sbdlabels
            finalmapping = None

        finalembeddings = self.embeddingdropout(finalembeddings)

        # SBD encoder and labels
        if self.sbdencodertype in ('lstm','gru'):
            feats, _ = self.sbdencoder(finalembeddings)
        else:
            feats = self.sbdposencoder(finalembeddings)
            feats = self.sbdencoder(feats)

        # Intermediate Feedforward layer
        feats = self.sbdfflayer(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)

        # logits for sbd
        sbdlogits = self.hidden2sbd(feats)

        #sbdlogits = sbdlogits.permute(0, 2, 1)

        # logits for pos
        #poslogits = self.hidden2postag(feats)
        #poslogits = poslogits.permute(0,2,1)

        del embeddings
        del finalembeddings
        del avgembeddings
        del feats

        torch.cuda.empty_cache()

        return sbdlogits,finallabels,finalmapping # returns the logits

class Tagger():
    def __init__(self,trainflag=False,trainfile=None,devfile=None,testfile=None,sbdrnndim=512,sbdrnnnumlayers=2,sbdrnnbidirectional=True,sbdrnndropout=0.3,sbdencodertype='lstm',sbdffdim=512,learningrate = 0.001):

        self.mtlmodel = MTLModel(sbdrnndim=sbdrnndim,sbdrnnnumlayers=sbdrnnnumlayers,sbdrnnbidirectional=sbdrnnbidirectional,sbdrnndropout=sbdrnndropout,sbdencodertype=sbdencodertype,sbdffdim=sbdffdim)

        if trainflag == True:

            from torch.utils.tensorboard import SummaryWriter
            if os.path.isdir('../data/tensorboarddir/'):
                shutil.rmtree('../data/tensorboarddir/')
            os.mkdir('../data/tensorboarddir/')

            if not os.path.isdir('../data/checkpoint/'):
                os.mkdir('../data/checkpoint/')

            self.writer = SummaryWriter('../data/tensorboarddir/')

            self.trainingdatafile = '../data/sentsplit_postag_train_gold.tab'
            self.devdatafile = '../data/sentsplit_postag_dev_gold.tab'
        else:
            self.testdatafile = '../data/sentsplit_postag_test_gold.tab'

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
                                           list(self.mtlmodel.hidden2sbd.parameters()), lr=learningrate)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[150,400],gamma=0.1)
        self.evalstep = 20

        self.set_seed(42)

    def set_seed(self, seed):

        random.seed(seed)
        torch.manual_seed(seed)

    def train(self):

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

        epochs = 1500

        trainingdata = read_file()
        devdata = read_file(mode='dev')

        for epoch in range(1,epochs):

            self.mtlmodel.train()
            self.optimizer.zero_grad()

            data = sample(trainingdata,SAMPLE_SIZE)
            data = [datum for datum in data if len(datum) == self.mtlmodel.sequence_length]
            self.mtlmodel.batch_size = len(data)

            sbdlogits, sbdlabels, badrecords = self.mtlmodel(data)
            sbdtags = torch.LongTensor(sbdlabels).to(self.device)

            lengths = [self.mtlmodel.sequence_length] * self.mtlmodel.batch_size
            lengths = torch.LongTensor(lengths).to(self.device)

            #scores = (poslogits,lengths,self.mtlmodel.poscrf.transitions)
            #sbdloss = self.sbdloss

            sbdlogits = sbdlogits.permute(0,2,1)
            sbdloss = self.sbdloss(sbdlogits,sbdtags)
            #posloss = self.postagloss(scores,postags)

            #mtlloss = posloss + sbdloss # uniform weighting. # TODO: learnable weights?
            #mtlloss.backward()
            sbdloss.backward()
            self.optimizer.step()
            #self.scheduler.step()

            #self.writer.add_scalar('train_pos_loss', posloss.item(), epoch)
            self.writer.add_scalar('train_sbd_loss', sbdloss.item(), epoch)
            #self.writer.add_scalar('train_joint_loss', mtlloss.item(), epoch)

            if epoch % self.evalstep == 0:

                self.mtlmodel.eval()
                #start = time()
                with torch.no_grad():

                    totaldevloss = 0
                    allpreds = []
                    allgold = []
                    invalidlabelscount = 0

                    for slice in devdata:

                        preds = []

                        goldlabels = [s.split('\t')[2].strip() for s in slice]
                        goldlabels = [self.mtlmodel.sbd_tag2idx[s] for s in goldlabels]

                        sbdlogits, sbdlabels, finalmapping = self.mtlmodel(slice,mode='dev')

                        if sbdlabels is not None:
                            # get the predictions - on non-shingled data
                            for idx in finalmapping:
                                snum, position = finalmapping[idx]
                                label = torch.argmax(sbdlogits[snum][position]).item()

                                preds.append(label)

                            # get the loss - on 'shingled' data
                            sbdlogits = sbdlogits.permute(0,2,1)
                            sbdtags = torch.LongTensor(sbdlabels).to(self.device)
                            devloss = self.sbdloss(sbdlogits, sbdtags).item()

                        else:
                            preds = [self.mtlmodel.sbd_tag2idx["O"] for _ in goldlabels]
                            invalidlabelscount += len(goldlabels)
                            devloss = 0

                        totaldevloss += devloss
                        allpreds.extend(preds)
                        allgold.extend(goldlabels)

                    #print ('dev inference')
                    #print (time() - start)

                    goldspans = []
                    predspans = []
                    goldstartindex = 0
                    predstartindex = 0
                    for i in range(0,len(allgold)):
                        if allgold[i] == 1: #B-SENT
                            goldspans.append(UDSpan(goldstartindex,i))
                            goldstartindex = i
                        if allpreds[i] == 1:
                            predspans.append(UDSpan(predstartindex,i))
                            predstartindex = i


                    scores = spans_score(goldspans,predspans)

                    print ('invalid labels:' + str(invalidlabelscount))

                    self.writer.add_scalar("dev_loss",round(totaldevloss/len(devdata),2),int(epoch / self.evalstep))
                    self.writer.add_scalar("dev_f1", round(scores.f1,2), int(epoch / self.evalstep))
                    self.writer.add_scalar("dev_precision", round(scores.precision, 2), int(epoch / self.evalstep))
                    self.writer.add_scalar("dev_recall", round(scores.recall, 2), int(epoch / self.evalstep))


                    print ('dev f1:' + str(scores.f1))
                    print('dev precision:' + str(scores.precision))
                    print('dev recall:' + str(scores.recall))
                    print ('\n')


    def predict(self):
        pass

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


def main(): # testing only

    iahltwikitrain = '/home/nitin/Desktop/IAHLT/UD_Hebrew-IAHLTwiki/he_iahltwiki-ud-train.conllu'
    iahltwikidev = '/home/nitin/Desktop/IAHLT/UD_Hebrew-IAHLTwiki/he_iahltwiki-ud-dev.conllu'


    tagger = Tagger(trainflag=True,trainfile=iahltwikitrain,devfile=iahltwikidev)
    tagger.prepare_data_files()
    tagger.train()

    print ('here')


if __name__ == "__main__":
    main()
