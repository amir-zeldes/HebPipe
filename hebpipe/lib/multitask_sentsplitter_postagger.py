import conllu
import torch
import torch.nn as nn
import os
import shutil
import flair
import random
import math
import gc

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from transformers import BertModel,BertTokenizerFast
from lib.allennlp.conditional_random_field import ConditionalRandomField
from lib.allennlp.time_distributed import TimeDistributed
from random import sample
from collections import defaultdict
from sklearn.metrics import f1_score, precision_score,recall_score

from time import time

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256,garbage_collection_threshold:0.2"


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
    def __init__(self,rnndim=512,rnnnumlayers=2,rnnbidirectional=True,rnndropout=0.3,encodertype='lstm',ffdim=512,batchsize=8,transformernumlayers=6,nhead=8,sequencelength=128):
        super(MTLModel,self).__init__()

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #self.device = 'cpu'
        #self.START_TAG = "<START>"
        #self.STOP_TAG = "<STOP>"

        self.postagset = {'ADJ':0, 'ADP':1, 'ADV':2, 'AUX':3, 'CCONJ':4, 'DET':5, 'INTJ':6, 'NOUN':7, 'NUM':8, 'PRON':9, 'PROPN':10, 'PUNCT':11, 'SCONJ':12, 'SYM':13, 'VERB':14, 'X':15} # derived from HTB and IAHLTWiki trainsets #TODO: add other UD tags?
        self.sbd_tag2idx = {'B-SENT': 1,'O': 0,}#self.START_TAG:2,self.STOP_TAG:3}

        self.sequence_length = sequencelength
        self.batch_size = batchsize
        self.encodertype = encodertype

        self.tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
        self.model = BertModel.from_pretrained('onlplab/alephbert-base').to(self.device)

        # Bi-LSTM Encoder
        self.embeddingdim = 768 * 1 # based on BERT model with Flair layers
        self.rnndim = rnndim
        self.rnnnumlayers = rnnnumlayers
        self.rnnbidirectional = rnnbidirectional
        self.rnndropout = rnndropout

        if encodertype == 'lstm':
            self.encoder = nn.LSTM(input_size=self.embeddingdim, hidden_size=self.rnndim // 2,
                                 num_layers=self.rnnnumlayers, bidirectional=self.rnnbidirectional,
                                 dropout=self.rnndropout,batch_first=True).to(self.device)
        elif encodertype == 'gru':
            self.encoder = nn.GRU(input_size=self.embeddingdim, hidden_size=self.rnndim // 2,
                                   num_layers=self.rnnnumlayers, bidirectional=self.rnnbidirectional,
                                   dropout=self.rnndropout,batch_first=True).to(self.device)
        elif self.encodertype == 'transformer':
            self.transformernumlayers = transformernumlayers
            self.nhead = nhead
            self.encoderlayer = nn.TransformerEncoderLayer(d_model= self.embeddingdim,nhead=nhead).to(self.device)
            self.encoder = nn.TransformerEncoder(self.encoderlayer,num_layers=self.transformernumlayers).to(self.device)
            self.posencoder = PositionalEncoding(d_model=self.embeddingdim).to(self.device)

        # param init
        for name, param in self.encoder.named_parameters():
            try:
                if 'bias' in name:
                    nn.init.constant_(param,0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
            except ValueError as ex:
                nn.init.constant_(param,0.0)

        self.relu = nn.ReLU()

        # Intermediate feedforward layer
        self.ffdim = ffdim
        if self.encodertype == 'transformer':
            self.fflayer = TimeDistributed(nn.Linear(in_features=self.embeddingdim, out_features=self.ffdim)).to(self.device)
        else:
            self.fflayer = TimeDistributed(nn.Linear(in_features=self.rnndim, out_features=self.ffdim)).to(self.device)

        # param init
        for name, param in self.fflayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Label space for the pos tagger
        # TODO: CRF?
        #self.hidden2postag = TimeDistributed(nn.Linear(in_features=self.ffdim,out_features=len(self.postagset.keys()))).to(self.device)

        # Label space for sent splitter
        self.hidden2sbd = TimeDistributed(nn.Linear(in_features=self.ffdim,out_features=len(self.sbd_tag2idx.keys())).to(self.device))

        # param init
        for name, param in self.hidden2sbd.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.sigmoid = nn.Sigmoid()

        #self.sbdcrf = ConditionalRandomField(len(self.sbd_tag2idx),include_start_end_transitions=False).to(self.device)


    def forward(self,data):

        """
        slice is a list of tuples of length = seq_len. Each tuple is (token, pos tag, sentence boundary label)
        """
        badrecords = []
        data = [d.split() for d in data] # for AlephBERT
        tokens = self.tokenizer(data,return_tensors='pt',padding=True,is_split_into_words=True).to(self.device) # tell AlephBERT that there is some tokenization already. Otherwise its own subword tokenization messes things up.

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

        avgembeddings = torch.stack(avgembeddings)

        #print ('average embeddings')
        #print (time() - start)

        if self.encodertype in ('lstm','gru'):
            feats, _ = self.encoder(avgembeddings)
        else:
            feats = self.posencoder(avgembeddings)
            feats = self.encoder(feats)


        # Intermediate Feedforward layer
        feats = self.fflayer(feats)
        feats = self.relu(feats)

        # logits for pos
        #poslogits = self.hidden2postag(feats)
        #poslogits = poslogits.permute(0,2,1)

        # logits for sbd
        sbdlogits = self.hidden2sbd(feats)
        #sbdlogits = torch.squeeze(sbdlogits)

        sbdlogits = sbdlogits.permute(0,2,1)

        #sbdloss = self.sbdcrf(sbdlogits, sbdtags)
        #viterbitags = self.sbdcrf.viterbi_tags(sbdlogits)

        #return None,sbdloss,viterbitags

        del embeddings
        del avgembeddings
        del feats

        gc.collect()
        torch.cuda.empty_cache()

        return sbdlogits,badrecords

class Tagger():
    def __init__(self,trainflag=False,trainfile=None,devfile=None,testfile=None,rnndim=512,rnnnumlayers=2,rnnbidirectional=True,rnndropout=0.3,encodertype='lstm',ffdim=512,learningrate = 0.0001):

        self.mtlmodel = MTLModel(rnndim,rnnnumlayers,rnnbidirectional,rnndropout,encodertype,ffdim)

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
        #self.device = 'cpu'

        self.trainflag = trainflag
        self.trainfile = trainfile
        self.devfile = devfile
        self.testfile = testfile

        self.learningrate = learningrate

        # Loss for pos tagging
        #self.postagloss = nn.CrossEntropyLoss()
        #self.postagloss.to(self.device)

        self.sbdloss = nn.CrossEntropyLoss() #(weight=torch.FloatTensor([1,5]))
        self.sbdloss.to(self.device)

        self.optimizer = torch.optim.Adam(list(self.mtlmodel.encoder.parameters()) +  list(self.mtlmodel.fflayer.parameters()) + list(self.mtlmodel.hidden2sbd.parameters()), lr=learningrate)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,milestones=[500,2500],gamma=0.01)
        self.evalstep = 20

        self.stride_size = 10

        self.set_seed(42)

    def set_seed(self, seed):

        random.seed(seed)
        torch.manual_seed(seed)


    def shingle_predict(self,toks,labels=None,type='sbd'):

        """
        Shingles data, then predicts the tag. Applies to dev and test sets only
        pass labels if they exist e.g for dev / test  Otherwise it's inference on new data.
        pass type for the type of label, sbd or pos
        """

        spans = []
        if labels:
            labelspans = []
        final_mapping = {}
        # Hack tokens up into overlapping shingles
        wraparound = toks[-self.stride_size:] + toks + toks[: self.mtlmodel.sequence_length]
        if labels:
            labelwraparound = labels[-self.stride_size:] + labels + labels[: self.mtlmodel.sequence_length]
        idx = 0
        mapping = defaultdict(set)
        snum = 0
        while idx < len(toks):
            if idx + self.mtlmodel.sequence_length < len(wraparound):
                span = wraparound[idx: idx + self.mtlmodel.sequence_length]
                if labels:
                    labelspan = labelwraparound[idx: idx + self.mtlmodel.sequence_length]
            else:
                span = wraparound[idx:]
                if labels:
                    labelspan = labelwraparound[idx:]
            sent = " ".join(span)
            spans.append(sent)
            if labels:
                labelspans.append(labelspan)

            for i in range(idx - self.stride_size, idx + self.mtlmodel.sequence_length - self.stride_size):
                # start, end, snum
                if i >= 0 and i < len(toks):
                    mapping[i].add((idx - self.stride_size, idx + self.mtlmodel.sequence_length - self.stride_size, snum))
            idx += self.stride_size
            snum += 1



        for idx in mapping:
            best = self.mtlmodel.sequence_length
            for m in mapping[idx]:
                start, end, snum = m
                dist_to_end = end - idx
                dist_to_start = idx - start
                delta = abs(dist_to_end - dist_to_start)
                if delta < best:
                    best = delta
                    final_mapping[idx] = (snum, idx - start)  # Get sentence number and position in sentence

        self.mtlmodel.batch_size = len(spans)

        sbdlogits,badrecords = self.mtlmodel(spans)
        predictions = torch.argmax(sbdlogits,dim=1)

        badrecords = sorted(badrecords,reverse=True)
        for record in badrecords:
            labelspans.pop(record)

        labelspans = torch.LongTensor(labelspans).to(self.device)

        labels = []
        for idx in final_mapping:
            snum, position = final_mapping[idx]
            label = predictions[snum][position].item()

            labels.append(label)

        loss = self.sbdloss(sbdlogits,labelspans)

        del sbdlogits
        del labelspans

        #gc.collect()
        torch.cuda.empty_cache()

        #print (torch.cuda.memory_stats("cuda:0"))


        return labels,loss.item()

    def train(self):

        def read_file(mode='train'):

            dataset = []
            if mode == 'dev':
                with open(self.devdatafile, 'r') as fi:
                    lines = fi.readlines()
                    lines = list(reversed(lines))  # hebrew is right to left...

                    # shingle it here to get more training data
                    for idx in range(0, len(lines), 128):
                        if idx + self.mtlmodel.sequence_length >= len(lines):
                            slice = lines[idx:len(lines)]
                        else:
                            slice = lines[idx: idx + self.mtlmodel.sequence_length]

                        dataset.append(slice)

            else:
                with open(self.trainingdatafile,'r') as fi:
                    lines = fi.readlines()
                    lines = list(reversed(lines)) # hebrew is right to left...

                    # shingle it here to get more training data
                    for idx in range(0,len(lines),self.stride_size):
                        if idx + self.mtlmodel.sequence_length >= len(lines):
                            slice = lines[idx:len(lines)]
                        else:
                            slice = lines[idx: idx + self.mtlmodel.sequence_length]

                        dataset.append(slice)

            return dataset

        epochs = 3000

        trainingdata = read_file()
        devdata = read_file(mode='dev')

        for epoch in range(1,epochs):

            self.mtlmodel.train()
            self.optimizer.zero_grad()

            data = sample(trainingdata,self.mtlmodel.batch_size)
            data = [datum for datum in data if len(datum) == self.mtlmodel.sequence_length]
            self.mtlmodel.batch_size = len(data)

            sents = [' '.join([s.split('\t')[0].strip() for s in sls]) for sls in data]

            sbdtags = [[s.split('\t')[2].strip() for s in sls] for sls in data]
            sbdtags = [[self.mtlmodel.sbd_tag2idx[t] for t in tag] for tag in sbdtags]

            sbdlogits,badrecords = self.mtlmodel(sents)
            badrecords = sorted(badrecords,reverse=True)

            for record in badrecords:
                sbdtags.pop(record)

            sbdtags = torch.LongTensor(sbdtags).to(self.device)
            sbdloss = self.sbdloss(sbdlogits,sbdtags)

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

                with torch.no_grad():

                    """
                    old_batch_size = self.mtlmodel.batch_size
                    data = [datum for datum in devdata if len(datum) == self.mtlmodel.sequence_length]
                    self.mtlmodel.batch_size = len(data)

                    sents = [' '.join([s.split('\t')[0].strip() for s in sls]) for sls in data]

                    sbdtags = [[s.split('\t')[2].strip() for s in sls] for sls in data]
                    sbdtags = [[self.mtlmodel.sbd_tag2idx[t] for t in tag] for tag in sbdtags]
                    goldlabels = [t for tags in sbdtags for t in tags]
                    sbdtags = torch.LongTensor(sbdtags).to(self.device)
                    """


                    totaldevloss = 0
                    allpreds = []
                    allgold = []
                    for slice in devdata:

                        sents = [s.split('\t')[0].strip() for s in slice]
                        goldlabels = [s.split('\t')[2].strip() for s in slice]
                        goldlabels = [self.mtlmodel.sbd_tag2idx[s] for s in goldlabels]

                        preds,devloss = self.shingle_predict(sents,goldlabels)
                        totaldevloss += devloss
                        allpreds.extend(preds)
                        allgold.extend(goldlabels)



                    """
                    sbdlogits = self.mtlmodel(sents)
                    devloss = self.sbdloss(sbdlogits,sbdtags)

                    preds = torch.argmax(sbdlogits,dim=1)
                    preds = preds.tolist()
                    preds = [p for pred in preds for p in pred]
                    """

                    f1 = f1_score(allgold,allpreds)
                    precision = precision_score(allgold,allpreds)
                    recall = recall_score(allgold,allpreds)

                    self.writer.add_scalar("dev_loss",round(totaldevloss,2),int(epoch / self.evalstep))
                    self.writer.add_scalar("dev_f1", round(f1,2), int(epoch / self.evalstep))
                    self.writer.add_scalar("dev_precision", round(precision, 2), int(epoch / self.evalstep))
                    self.writer.add_scalar("dev_recall", round(recall, 2), int(epoch / self.evalstep))


                    print ('dev f1:' + str(f1))
                    print('dev precision:' + str(precision))
                    print('dev recall:' + str(recall))

                torch.cuda.empty_cache()
                self.mtlmodel.train()
                #self.mtlmodel.batch_size = old_batch_size


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
                    for i in range(0,len(sent)): # This will disregard the final punct in each sentence.

                        if isinstance(sent[i]['id'],tuple): continue

                        if i == len(sent) - 2 and (sent[i + 1]['form'] == '.' and sent[i + 1]['upos'] == 'PUNCT'):
                            tr.write(sent[i]['form'] + '\t' + sent[i]['upos'] + '\t' + 'B-SENT' + '\n')
                        elif i == len(sent) - 1 and (sent[i]['form'] != '.' and sent[i]['upos'] != 'PUNCT'):
                            tr.write(sent[i]['form'] + '\t' + sent[i]['upos'] + '\t' + 'B-SENT' + '\n')
                        elif i != len(sent) - 1:
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
