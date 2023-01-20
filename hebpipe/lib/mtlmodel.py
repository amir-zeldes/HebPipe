import conllu
import torch
import torch.nn as nn
import os
import shutil
import random
import re
import argparse


from flair.data import Dictionary, Sentence
from transformers import BertModel,BertTokenizerFast,BertConfig
from random import sample
from collections import defaultdict
try:
    from lib.dropout import WordDropout,LockedDropout
    from lib.crfutils.crf import CRF
    from lib.crfutils.viterbi import ViterbiDecoder,ViterbiLoss
    from lib.reorder_sgml import reorder
    from lib.tt2conll import conllize
except ModuleNotFoundError:
    from .dropout import WordDropout, LockedDropout
    from .crfutils.crf import CRF
    from .crfutils.viterbi import ViterbiDecoder, ViterbiLoss
    from .reorder_sgml import reorder
    from .tt2conll import conllize

from time import time

#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:256"

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


class MTLModel(nn.Module):
    def __init__(self,sbdrnndim=256,posrnndim=512,morphrnndim=512,sbdrnnnumlayers=1,posrnnnumlayers=1,morphrnnnumlayers=1,posfflayerdim=512,morphfflayerdim=512,sbdrnnbidirectional=True,posrnnbidirectional=True,morphrnnbidirectional=True,sbdencodertype='lstm',sbdfflayerdim=256,posencodertype='lstm',morphencodertype='lstm',batchsize=32,sequencelength=256,dropout=0.0,wordropout=0.05,lockeddropout=0.5,cpu=False):
        super(MTLModel,self).__init__()

        if cpu == False:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        # tagsets - amend labels here
        self.postagset = {'ADJ':0, 'ADP':1, 'ADV':2, 'AUX':3, 'CCONJ':4, 'DET':5, 'INTJ':6, 'NOUN':7, 'NUM':8, 'PRON':9, 'PROPN':10, 'PUNCT':11, 'SCONJ':12, 'SYM':13, 'VERB':14, 'X':15} # derived from HTB and IAHLTWiki trainsets #TODO: add other UD tags?
        self.sbd_tag2idx = {'B-SENT': 1,'O': 0}
        self.supertokenset = {'O':0,'B':1,'I':2,'E':3}

        # POS tagset in Dictionary object for Flair CRF
        self.postagsetcrf = Dictionary()
        for key in self.postagset.keys():
            self.postagsetcrf.add_item(key.strip())
        self.postagsetcrf.add_item("<START>")
        self.postagsetcrf.add_item("<STOP>")

        # FEATS dictionary
        # IMPORTANT: This should be sorted by key
        self.featstagset = {'Definite=Com':0, 'Definite=Cons':1, 'Definite=Def':2, 'Definite=Ind':3, 'Definite=Spec':4,
                            'Gender=Fem':5, 'Gender=Masc':6, 'HebBinyan=HIFIL':7, 'HebBinyan=HITPAEL':8, 'HebBinyan=HUFAL':9, 'HebBinyan=NIFAL':10,
                            'HebBinyan=NITPAEL':11, 'HebBinyan=PAAL':12, 'HebBinyan=PIEL':13, 'HebBinyan=PUAL':14, 'Number=Dual':15, 'Number=Plur':16, 'Number=Sing':17,
                            'Tense=Fut':18, 'Tense=Past':19, 'Tense=Pres':20,'VerbForm=Inf':21, 'VerbForm=Part':22,'Voice=Act':23, 'Voice=Mid':24, 'Voice=Pass':25}
        self.idxtofeatstagset = {v: k for k, v in self.featstagset.items()}
        # shared hyper-parameters
        self.sequence_length = sequencelength
        self.batch_size = batchsize

        # Embedding parameters and model
        config = BertConfig.from_pretrained('onlplab/alephbert-base',output_hidden_states=True)
        self.tokenizer = BertTokenizerFast.from_pretrained('onlplab/alephbert-base')
        self.model = BertModel.from_pretrained('onlplab/alephbert-base',config=config).to(self.device)
        self.embeddingdim = 768
        self.lastn = 4

        for param in self.model.base_model.parameters():
            param.requires_grad = False

        # Bi-LSTM Encoder for SBD
        self.sbdrnndim = sbdrnndim
        self.sbdrnnnumlayers = sbdrnnnumlayers
        self.sbdrnnbidirectional = sbdrnnbidirectional

        #Bi-LSTM Encoder for POS tagging
        self.posrnndim = posrnndim
        self.posrnnnumlayers = posrnnnumlayers
        self.posrnnbidirectional = posrnnbidirectional

        # Encoder for feats
        self.morphrnndim = morphrnndim
        self.morphrnnnumlayers = morphrnnnumlayers
        self.morphrnnbidirectional = morphrnnbidirectional

        if sbdencodertype == 'lstm':
            self.sbdencoder = nn.LSTM(input_size=self.embeddingdim, hidden_size=self.sbdrnndim // 2,
                                 num_layers=self.sbdrnnnumlayers, bidirectional=self.sbdrnnbidirectional,
                                 batch_first=True).to(self.device)
        elif sbdencodertype == 'gru':
            self.sbdencoder = nn.GRU(input_size=self.embeddingdim, hidden_size=self.sbdrnndim // 2,
                                   num_layers=self.sbdrnnnumlayers, bidirectional=self.sbdrnnbidirectional,
                                   batch_first=True).to(self.device)

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
            self.posencoder = nn.LSTM(input_size=self.embeddingdim +  len(self.supertokenset) + 1, hidden_size=self.posrnndim // 2,
                                 num_layers=self.posrnnnumlayers, bidirectional=self.posrnnbidirectional,
                                 batch_first=True).to(self.device)
        elif posencodertype == 'gru':
            self.posencoder = nn.GRU(input_size=self.embeddingdim +  len(self.supertokenset) + 1 , hidden_size=self.posrnndim // 2,
                                   num_layers=self.posrnnnumlayers, bidirectional=self.posrnnbidirectional,
                                   batch_first=True).to(self.device)

        # param init
        for name, param in self.posencoder.named_parameters():
            try:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
            except ValueError as ex:
                nn.init.constant_(param, 0.0)

        if morphencodertype == 'lstm':
            self.morphencoder = nn.LSTM(input_size=self.embeddingdim + len(self.postagsetcrf) + len(self.supertokenset) + 1, hidden_size=self.morphrnndim // 2,
                                 num_layers=self.morphrnnnumlayers, bidirectional=self.morphrnnbidirectional,
                                 batch_first=True).to(self.device)
        elif morphencodertype == 'gru':
            self.morphencoder = nn.GRU(input_size=self.embeddingdim + len(self.postagsetcrf) + len(self.supertokenset) + 1, hidden_size=self.morphrnndim // 2,
                                   num_layers=self.morphrnnnumlayers, bidirectional=self.morphrnnbidirectional,
                                   batch_first=True).to(self.device)

        # param init
        for name, param in self.morphencoder.named_parameters():
            try:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_uniform_(param)
            except ValueError as ex:
                nn.init.constant_(param, 0.0)


        self.relu = nn.ReLU()

        # Reproject embeddings layer
        self.sbdembedding2nn = nn.Linear(in_features=self.embeddingdim ,out_features=self.embeddingdim).to(self.device)
        self.sbdfflayerdim = sbdfflayerdim
        self.sbdfflayer = nn.Linear(in_features=self.sbdrnndim, out_features=self.sbdfflayerdim).to(self.device)

        # param init
        for name, param in self.sbdembedding2nn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.sbdfflayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


        # Intermediate feedforward layer
        self.posembedding2nn = nn.Linear(in_features=self.embeddingdim + len(self.supertokenset) + 1,out_features=self.embeddingdim  +  len(self.supertokenset) + 1 ).to(self.device)
        self.posfflayerdim = posfflayerdim
        self.posfflayer = nn.Linear(in_features=self.posrnndim, out_features=self.posfflayerdim).to(self.device)

        # param init
        for name, param in self.posembedding2nn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        for name, param in self.posfflayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Intermediate feedforward layer
        self.morphembedding2nn = nn.Linear(in_features=self.embeddingdim + len(self.postagsetcrf) + len(self.supertokenset) + 1 ,
                                         out_features=self.embeddingdim + len(self.postagsetcrf) + len(self.supertokenset) + 1).to(self.device)
        self.morphfflayerdim = morphfflayerdim
        self.morphfflayer = nn.Linear(in_features=self.morphrnndim, out_features=self.morphfflayerdim).to(self.device)

        # param init
        for name, param in self.morphembedding2nn.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # param init
        for name, param in self.morphfflayer.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Label space for the pos tagger
        self.hidden2postag = nn.Linear(in_features=self.posfflayerdim,out_features=len(self.postagsetcrf)).to(self.device)
        for name, param in self.hidden2postag.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        # Label space for sent splitter
        self.hidden2sbd = nn.Linear(in_features=self.sbdfflayerdim,out_features=len(self.sbd_tag2idx.keys())).to(self.device)
        # param init
        for name, param in self.hidden2sbd.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)


        # label space for morph feats
        self.hidden2feats = nn.Linear(in_features=self.morphfflayerdim,out_features=len(self.featstagset)).to(self.device)
        for name, param in self.hidden2feats.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.dropout = nn.Dropout(dropout)
        self.worddropout = WordDropout(wordropout)
        self.lockeddropout = LockedDropout(lockeddropout)

        self.poscrf = CRF(self.postagsetcrf,len(self.postagsetcrf),init_from_state_dict=False,cpu=cpu).to(self.device) # TODO: parameterize
        self.viterbidecoder = ViterbiDecoder(self.postagsetcrf,cpu=cpu)

        for name, param in self.poscrf.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.xavier_normal_(param)

        self.stride_size = 10
        self.sigmoid = nn.Sigmoid()

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
        featslabels = None

        # Extract the sentences and labels
        if mode == 'train': # training is on a batch, so 3D tensor
            sentences = [' '.join([s.split('\t')[0].strip() for s in sls]) for sls in data]
            sbdlabels = [[self.sbd_tag2idx[s.split('\t')[2].strip()] for s in sls] for sls in data]
            poslabels = [[self.postagsetcrf.get_idx_for_item(s.split('\t')[1].strip()) for s in sls] for sls in data]

            supertokenlabels = []
            featslabels = []

            for sls in data:
                record = []
                featsrecord = []
                for s in sls:
                    temp = [0] * len(self.supertokenset)
                    temp[self.supertokenset[s.split('\t')[3].strip()]] = 1
                    record.append(temp)

                    tempfeats = [0] * len(self.featstagset)
                    fts = s.split('\t')[-1].strip()
                    if fts != '':
                        fts = fts.split('|')
                        for f in fts:
                            key = f.split('=')[0]
                            value = f.split('=')[1]
                            if ',' not in value:
                                if f not in self.featstagset.keys(): continue
                                tempfeats[self.featstagset[f]] = 1
                            else:
                                value = value.split(',')
                                for v in value:
                                    if key + '=' + v not in self.featstagset.keys(): continue
                                    tempfeats[self.featstagset[key + '=' + v]] = 1

                    featsrecord.append(tempfeats)

                supertokenlabels.append(record)
                featslabels.append(featsrecord)

        elif mode == 'dev': # inference is on a single record
            sentences = [' '.join([s.split('\t')[0].strip() for s in data])]
            sbdlabels = [self.sbd_tag2idx[s.split('\t')[2].strip()] for s in data]
            poslabels = [self.postagsetcrf.get_idx_for_item(s.split('\t')[1].strip()) for s in data]

            supertokenlabels = []
            featslabels = []
            for s in data:
                temp = [0] * len(self.supertokenset)
                temp[self.supertokenset[s.split('\t')[3].strip()]] = 1
                supertokenlabels.append(temp)

                tempfeats = [0] * len(self.featstagset)
                fts = s.split('\t')[-1].strip()
                if fts != '':
                    fts = fts.split('|')
                    for f in fts:
                        key = f.split('=')[0]
                        value = f.split('=')[1]
                        if ',' not in value:
                            if f not in self.featstagset.keys(): continue
                            tempfeats[self.featstagset[f]] = 1
                        else:
                            value = value.split(',')
                            for v in value:
                                if key + '=' + v not in self.featstagset.keys(): continue
                                tempfeats[self.featstagset[key + '=' + v]] = 1

                featslabels.append(tempfeats)

        else: # test - a tuple of text and supertoken labels
            sentences = [' '.join([s.split('\t')[0].strip() for s in data[0]])]

            supertokenlabels = []
            for s in data[1]:
                temp = [0] * len(self.supertokenset)
                temp[self.supertokenset[s.strip()]] = 1
                supertokenlabels.append(temp)

            sbdlabels = None
            poslabels = None

        # Make embeddings and scalar average them across subwords, vertically.
        sentences = [d.split() for d in sentences] # for AlephBERT
        tokens = self.tokenizer(sentences,return_tensors='pt',padding=True,is_split_into_words=True,truncation=True).to(self.device) # tell AlephBERT that there is some tokenization already.

        output = self.model(**tokens)
        hiddenstates = output[2][-self.lastn:]
        scalarsum = hiddenstates[0]
        for i in range(1,self.lastn):
            scalarsum = torch.add(scalarsum,hiddenstates[i],alpha=1)

        embeddings = torch.div(scalarsum,self.lastn)
        #embeddings = embeddings.to(self.device)

        """
        Average the subword embeddings within the horizontal sequence.
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
                    emb.append(torch.zeros(self.embeddingdim ,device=self.device))
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

        sbdembeddings = self.dropout(sbdembeddings)
        sbdembeddings = self.worddropout(sbdembeddings)
        sbdembeddings = self.lockeddropout(sbdembeddings)

        sbdembeddings = self.sbdembedding2nn(sbdembeddings)

        # SBD encoder and labels
        feats, _ = self.sbdencoder(sbdembeddings)
        feats = self.sbdfflayer(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        feats = self.lockeddropout(feats)

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

        supertokenlabels = torch.LongTensor(supertokenlabels)
        supertokenlabels = supertokenlabels.to(self.device)
        if mode in ('dev','test'):
            supertokenlabels = torch.unsqueeze(supertokenlabels,dim=0)

        # Add the SBD predictions to the POS Encoder Input!
        posembeddings = torch.cat((avgembeddings,sbdpreds,supertokenlabels),dim=2)

        posembeddings = self.dropout(posembeddings)
        posembeddings = self.worddropout(posembeddings)
        posembeddings = self.lockeddropout(posembeddings)
        posembeddings = self.posembedding2nn(posembeddings)

        feats,_ = self.posencoder(posembeddings)
        feats = self.posfflayer(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        feats = self.lockeddropout(feats)

        # logits for pos
        poslogits = self.hidden2postag(feats)
        poslogits = self.poscrf(poslogits)

        # get the pos CRF predictions
        if mode == 'train':
            lengths = [self.sequence_length] * self.batch_size
        else:
            lengths = [self.sequence_length]
        scores = (poslogits, lengths, self.poscrf.transitions)
        sents = []
        for s in sentences:
            sents.append(Sentence(' '.join(s),use_tokenizer=False))

        pospreds = self.viterbidecoder.decode(scores, False, sents)
        pospreds = [[self.postagsetcrf.get_idx_for_item(p[0])for p in pr] for pr in pospreds[0]]
        pospredsonehot = []
        for pred in pospreds:
            preds = []
            for p in pred:
                onehot = [0] * len(self.postagsetcrf)
                onehot[p] = 1
                preds.append(onehot)
            pospredsonehot.append(preds)

        pospredsonehot = torch.LongTensor(pospredsonehot)
        pospredsonehot = pospredsonehot.to(self.device)

        morphembeddings = torch.cat((avgembeddings, sbdpreds, supertokenlabels,pospredsonehot), dim=2)
        morphembeddings = self.dropout(morphembeddings)
        morphembeddings = self.worddropout(morphembeddings)
        morphembeddings = self.lockeddropout(morphembeddings)
        morphembeddings = self.morphembedding2nn(morphembeddings)

        feats, _ = self.morphencoder(morphembeddings)
        feats = self.morphfflayer(feats)
        feats = self.relu(feats)
        feats = self.dropout(feats)
        feats = self.lockeddropout(feats)

        # logits for morphs
        featslogits = self.hidden2feats(feats)

        if mode in ('dev','test'):
            # Squeeze these to return to the Trainer for scores, now that we are done with them
            sbdpreds = torch.squeeze(sbdpreds,dim=2)
            sbdpreds = torch.squeeze(sbdpreds, dim=0)
            sbdpreds = sbdpreds.tolist()

            # Unroll the pos predictions
            pospreds = [p for pred in pospreds for p in pred]

        else:
            sbdpreds = None
            pospreds = None

        return sbdlogits, finalsbdlabels, sbdpreds, poslogits, poslabels, pospreds, featslogits,featslabels # returns the logits and labels

class Tagger():
    def __init__(self,trainflag=False,trainfile=None,devfile=None,testfile=None,sbdrnndim=256,sbdrnnnumlayers=1,sbdrnnbidirectional=True,sbdfflayerdim=256,posrnndim=512,posrnnnumlayers=1,posrnnbidirectional=True,posfflayerdim=512,morphrnndim=512,morphrnnnumlayers=1,morphrnnbidirectional=True,morphfflayerdim=512,morphencodertype='lstm',dropout=0.05,wordropout=0.05,lockeddropout=0.5,sbdencodertype='lstm',posencodertype='lstm',learningrate = 0.001,bestmodelpath='../data/checkpoint/',batchsize=32,sequencelength=256,datatype='htb',cpu=False):

        self.mtlmodel = MTLModel(sbdrnndim=sbdrnndim,sbdrnnnumlayers=sbdrnnnumlayers,sbdrnnbidirectional=sbdrnnbidirectional,sbdencodertype=sbdencodertype,sbdfflayerdim=sbdfflayerdim,dropout=dropout,wordropout=wordropout,lockeddropout=lockeddropout,posrnndim=posrnndim,posrnnbidirectional=posrnnbidirectional,posencodertype=posencodertype,posrnnnumlayers=posrnnnumlayers,posfflayerdim=posfflayerdim,morphrnndim=morphrnndim,morphrnnnumlayers=morphrnnnumlayers,morphencodertype=morphencodertype,morphrnnbidirectional=morphrnnbidirectional,morphfflayerdim=morphfflayerdim,batchsize=batchsize,sequencelength=sequencelength,cpu=cpu)

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

            self.bestmodel = bestmodelpath + datatype + '_best_mtlmodel.pt'

        if cpu == False:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = "cpu"

        self.trainflag = trainflag
        self.trainfile = trainfile
        self.devfile = devfile
        self.testfile = testfile

        self.learningrate = learningrate

        # Loss for pos tagging
        self.postagloss = ViterbiLoss(self.mtlmodel.postagsetcrf,cpu=cpu)
        self.postagloss.to(self.device)

        # Loss for sentence splitting
        self.sbdloss = nn.CrossEntropyLoss(weight=torch.FloatTensor([1,3]))
        self.sbdloss.to(self.device)

        self.featsloss = nn.BCEWithLogitsLoss()
        self.featsloss.to(self.device)

        self.optimizer = torch.optim.AdamW(list(self.mtlmodel.sbdencoder.parameters()) +  list(self.mtlmodel.sbdembedding2nn.parameters()) +
                                           list(self.mtlmodel.hidden2sbd.parameters()) + list(self.mtlmodel.posencoder.parameters()) + list(self.mtlmodel.posembedding2nn.parameters())
                                           + list(self.mtlmodel.hidden2postag.parameters()) + list(self.mtlmodel.poscrf.parameters())
                                           + list(self.mtlmodel.hidden2feats.parameters()) + list(self.mtlmodel.morphfflayer.parameters()) + list(self.mtlmodel.morphembedding2nn.parameters()) + list(self.mtlmodel.morphencoder.parameters())
                                           + list(self.mtlmodel.posfflayer.parameters()) + list(self.mtlmodel.sbdfflayer.parameters()), lr=learningrate)

        self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer,base_lr=learningrate/10,max_lr=learningrate,step_size_up=250,cycle_momentum=False)
        self.evalstep = 50

        self.sigmoidthreshold = 0.5

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

        epochs = 5000
        bestf1 = float('-inf')

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

            sbdlogits, sbdlabels, _, poslogits, poslabels, _ , featslogits, featslabels = self.mtlmodel(data)

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

            featstags = torch.FloatTensor(featslabels).to(self.device)
            featsloss = self.featsloss(featslogits,featstags)

            mtlloss = posloss + sbdloss + featsloss # TODO: learnable weights?
            mtlloss.backward()
            self.optimizer.step()
            self.scheduler.step()

            if old_batchsize != self.mtlmodel.batch_size:
                self.mtlmodel.batch_size = old_batchsize

            self.writer.add_scalar('train_pos_loss', posloss.item(), epoch)
            self.writer.add_scalar('train_sbd_loss', sbdloss.item(), epoch)
            self.writer.add_scalar('train_feats_loss', featsloss.item(), epoch)
            self.writer.add_scalar('train_joint_loss', mtlloss.item(), epoch)

            """""""""""""""""""""""""""""""""""""""""""""
            Do dev evaluation after evalstep number of epochs
            """""""""""""""""""""""""""""""""""""""""""""
            if epoch % self.evalstep == 0:

                self.mtlmodel.eval()

                with torch.no_grad():

                    totalsbddevloss = 0
                    totalposdevloss = 0
                    totalfeatsdevloss = 0

                    allsbdpreds = []
                    allsbdgold = []
                    allpospreds = []
                    allposgold = []
                    allfeatsgold = []
                    allfeatspreds = []

                    # because of shingling for SBD, the dev data needs to be split in slices for inference, as GPU may run out of memory with shingles on the full token list.
                    # shingling and SBD prediction is done on the individual slice, as well as POS tag predictions and feats.
                    # TODO This naturally increases prediction time...but can't think of a better way.
                    for slice in devdata:

                        old_seqlen = self.mtlmodel.sequence_length
                        if len(slice) != self.mtlmodel.sequence_length: # this will happen in one case, for the last slice in the dev batch
                            self.mtlmodel.sequence_length = len(slice)

                        # Extract gold labels
                        goldsbdlabels = [s.split('\t')[2].strip() for s in slice]
                        goldsbdlabels = [self.mtlmodel.sbd_tag2idx[s] for s in goldsbdlabels]
                        goldposlabels = [s.split('\t')[1].strip() for s in slice]
                        goldposlabels = [self.mtlmodel.postagsetcrf.get_idx_for_item(s) for s in goldposlabels]
                        goldfeatslabels = [s.split('\t')[4].strip() for s in slice]

                        # RUn through the model and get the labels and logits (and preds for stepwise models)
                        sbdlogits, sbdlabels, sbdpreds, poslogits, poslabels, pospreds, featslogits, featslabels = self.mtlmodel(slice,mode='dev')

                        # get the feats predictions
                        featspreds = self.mtlmodel.sigmoid(featslogits)
                        featspreds = (featspreds > self.sigmoidthreshold).long()
                        featspreds = torch.squeeze(featspreds).tolist()

                        # get the sbd loss
                        sbdlogits = sbdlogits.permute(0,2,1)
                        sbdtags = torch.LongTensor(sbdlabels).to(self.device)
                        sbddevloss = self.sbdloss(sbdlogits, sbdtags).item()

                        # get the pos loss
                        postags = torch.LongTensor(poslabels)
                        postags = postags.to(self.device)
                        lengths = [self.mtlmodel.sequence_length]
                        lengths = torch.LongTensor(lengths).to(self.device)
                        scores = (poslogits, lengths, self.mtlmodel.poscrf.transitions)
                        posdevloss = self.postagloss(scores,postags).item()

                        # get the feats loss
                        featstags = torch.FloatTensor(featslabels)
                        featstags = featstags.to(self.device)
                        featstags = torch.unsqueeze(featstags,dim=0)
                        featsdevloss = self.featsloss(featslogits,featstags).item()

                        # add up the losses across the slices for the avg
                        totalsbddevloss += sbddevloss
                        totalposdevloss += posdevloss
                        totalfeatsdevloss += featsdevloss

                        # build the feats tags for the sequence
                        featsslicepreds = []
                        for preds in featspreds:
                            featsstr = ''
                            for i in range(0,len(preds)):
                                if preds[i] != 0:
                                    if self.mtlmodel.idxtofeatstagset[i].split('=')[0] == featsstr.split('|')[-1].split('=')[0]:
                                        featsstr += ',' + self.mtlmodel.idxtofeatstagset[i].split('=')[1]
                                    else:
                                        if featsstr != '':
                                            featsstr = featsstr + '|' + self.mtlmodel.idxtofeatstagset[i]
                                        else:
                                            featsstr += self.mtlmodel.idxtofeatstagset[i]

                            featsslicepreds.append(featsstr)

                        # build the gold and predictions for the entire dev set
                        allsbdpreds.extend(sbdpreds)
                        allsbdgold.extend(goldsbdlabels)
                        allpospreds.extend(pospreds)
                        allposgold.extend(goldposlabels)
                        allfeatsgold.extend(goldfeatslabels)
                        allfeatspreds.extend(featsslicepreds)

                    #print ('inference time')
                    #print (time() - start)
                    if self.mtlmodel.sequence_length != old_seqlen:
                        self.mtlmodel.sequence_length = old_seqlen

                    # Now get the scores
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

                    correctfeats = sum([1 if p == g else 0 for p,g in zip(allfeatspreds,allfeatsgold)])
                    featsscores = Score(len(allfeatsgold),len(allfeatspreds),correctfeats,len(allfeatspreds))

                    # Write the scores and losses to tensorboard and console
                    mtlloss = (totalsbddevloss + totalposdevloss + totalfeatsdevloss) / len(devdata)

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

                    self.writer.add_scalar("feats_dev_loss",round(totalfeatsdevloss / len(devdata),4), int(epoch / self.evalstep))
                    self.writer.add_scalar("feats_dev_f1",round(featsscores.f1,4),int(epoch / self.evalstep))
                    self.writer.add_scalar("feats_dev_precision",round(featsscores.precision,4),int(epoch / self.evalstep))
                    self.writer.add_scalar("feats_dev_recall", round(featsscores.recall, 4),int(epoch / self.evalstep))

                    print ('sbd dev f1:' + str(sbdscores.f1))
                    print('sbd dev precision:' + str(sbdscores.precision))
                    print('sbd dev recall:' + str(sbdscores.recall))
                    print ('\n')

                    print('pos dev f1:' + str(posscores.f1))
                    print('pos dev precision:' + str(posscores.precision))
                    print('pos dev recall:' + str(posscores.recall))
                    print('\n')

                    print('feats dev f1:' + str(featsscores.f1))
                    print('feats dev precision:' + str(featsscores.precision))
                    print('feats dev recall:' + str(featsscores.recall))
                    print('\n')

                    # save the best model
                    if (sbdscores.f1 + posscores.f1 + featsscores.f1) / 3 > bestf1:
                        bestf1 = (sbdscores.f1 + posscores.f1 + featsscores.f1) / 3
                        bestmodel = self.bestmodel.replace('.pt','_' + str(round(mtlloss,6)) + '_' + str(round(sbdscores.f1,6)) + '_' + str(round(posscores.f1,6)) + '_' + str(round(featsscores.f1,6)) + '.pt')
                        torch.save({'epoch':epoch,'model_state_dict':self.mtlmodel.state_dict(),'optimizer_state_dict':self.optimizer.state_dict(),'poscrf_state_dict':self.mtlmodel.poscrf.state_dict()},bestmodel)

    def inference(self,toks,sent_tag='auto',checkpointfile=None):

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

        taggedsbdpreds = []

        # add super token tags
        supertokenlabels = []
        for i in range(0,len(lines)):
            if i > 0:
                prevtoken = lines[i-1]
            if i < len(lines) - 1:
                nexttoken = lines[i + 1]

            currtoken = lines[i]

            if is_tok(currtoken):
                if not is_tok(prevtoken):
                    if not is_tok(nexttoken):
                        supertokenlabels.append("O")
                    else:
                        supertokenlabels.append("B")
                else:
                    if not is_tok(nexttoken):
                        supertokenlabels.append("E")
                    else:
                        supertokenlabels.append("I")

                if sent_tag != 'auto':
                    if i != 2 and lines[i - 2] == "<" + sent_tag + ">":
                        taggedsbdpreds.append(1)
                    else:
                        taggedsbdpreds.append(0)

        toks = [l for l in lines if is_tok(l)]
        toks = [re.sub(r"\t.*", "", t) for t in toks]

        assert len(toks) == len(supertokenlabels)
        if sent_tag != 'auto':
            assert len(taggedsbdpreds) == len(toks)

        # slice up the token list into slices of seqlen for GPU RAM reasons
        for idx in range(0, len(toks), self.mtlmodel.sequence_length):
            if idx + self.mtlmodel.sequence_length >= len(toks):
                slice = toks[idx:len(toks)]
                supertokenslice = supertokenlabels[idx:len(toks)]
            else:
                slice = toks[idx: idx + self.mtlmodel.sequence_length]
                supertokenslice = supertokenlabels[idx: idx + self.mtlmodel.sequence_length]

            slices.append((slice,supertokenslice))

        test = [d for s in slices for d in s[0]]

        assert len(test) == len(toks)

        if checkpointfile is not None:

            checkpoint = torch.load(checkpointfile,map_location=self.device)
            self.mtlmodel.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.mtlmodel.poscrf.load_state_dict(checkpoint['poscrf_state_dict'])

        self.mtlmodel.eval()

        with torch.no_grad():

            allwords = []
            allsbdpreds = []
            allpospreds = []
            allfeatspreds = []

            for slice in slices:

                if len(slice[0]) != self.mtlmodel.sequence_length:  # this will happen in one case, for the last slice in the batch
                    self.mtlmodel.sequence_length = len(slice[0])

                _, _, sbdpreds, _,_,pospreds, featslogits,_ = self.mtlmodel(slice, mode='test')

                # get the feats predictions
                featspreds = self.mtlmodel.sigmoid(featslogits)
                featspreds = (featspreds > self.sigmoidthreshold).long()
                featspreds = torch.squeeze(featspreds).tolist()

                featsslicepreds = []
                for preds in featspreds:
                    featsstr = ''
                    for i in range(0, len(preds)):
                        if preds[i] != 0:
                            if self.mtlmodel.idxtofeatstagset[i].split('=')[0] == featsstr.split('|')[-1].split('=')[0]:
                                featsstr += ',' + self.mtlmodel.idxtofeatstagset[i].split('=')[1]
                            else:
                                if featsstr != '':
                                    featsstr = featsstr + '|' + self.mtlmodel.idxtofeatstagset[i]
                                else:
                                    featsstr += self.mtlmodel.idxtofeatstagset[i]

                    featsslicepreds.append(featsstr)

                allsbdpreds.extend(sbdpreds)
                allpospreds.extend(pospreds)
                allfeatspreds.extend(featsslicepreds)
                allwords.extend([s.split('\t')[0].strip() for s in slice[0]])

        allpospreds = [self.mtlmodel.postagsetcrf.get_item_for_index(p) for p in allpospreds]


        if sent_tag != 'auto':
            allsbdpreds = taggedsbdpreds

        return allsbdpreds,allpospreds, allfeatspreds, allwords

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
                length = -1
                for sent in data:
                    i = 0
                    while i < len(sent):
                        if isinstance(sent[i]['id'], tuple):
                            # fetch the super token tag
                            supertoken = 'B'
                            length = sent[i]['id'][-1] - sent[i]['id'][0]
                            start = sent[i]['id'][0]
                            i += 1
                            continue
                        elif length > 0 and supertoken in ('B','I'):
                            if sent[i]['id'] == start:
                                supertoken = 'B'
                            else:
                                supertoken = 'I'
                            length -=1
                        elif length == 0:
                            supertoken = 'E'
                            length = -1
                        elif length == -1:
                            supertoken = 'O'

                        if sent[i]['feats'] is not None:
                            feats = '|'.join(k + '=' + v for k,v in sent[i]['feats'].items())
                        else:
                            feats = ''

                        if sent[i]['id'] == 1:
                            tr.write(sent[i]['form'] + '\t' + sent[i]['upos'] + '\t' + 'B-SENT' + '\t' + supertoken + '\t' + feats + '\n')

                        else:
                            tr.write(sent[i]['form'] + '\t' + sent[i]['upos'] + '\t' + 'O' + '\t' + supertoken + '\t' + feats + '\n')

                        i += 1

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

    def predict(self, xml_data,out_mode='conllu',sent_tag='auto',checkpointfile = None):

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
        split_indices, pos_tags, morphs, words = self.inference(no_pos_lemma,sent_tag=sent_tag,checkpointfile=checkpointfile)

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
        data = lines.split("\n")
        retokenized = []
        for line in data:
            if line == "|":
                retokenized.append(line)
            else:
                retokenized.append("\n".join(line.split("|")))
        data = "\n".join(retokenized)

        """
        Now add the pos tags
        """
        bound_group_map = get_bound_group_map(data) if out_mode == "conllu" else None
        data = conllize(data, element="s", super_mapping=bound_group_map, attrs_as_comments=True)
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

        morphs = [m if m != '' else '_' for m in morphs]

        return "\n".join(output), lines, morphs, words

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

    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--seqlen', type=int, default=256)
    parser.add_argument('--trainbatch', type=int, default=32)
    parser.add_argument('--datatype', type=str, default='wiki')
    parser.add_argument('--sbdrnndim', type=int, default=256)
    parser.add_argument('--posrnndim', type=int, default=512)
    parser.add_argument('--morphrnndim', type=int, default=512)
    parser.add_argument('--sbdfflayerdim', type=int, default=256)
    parser.add_argument('--posfflayerdim', type=int, default=512)
    parser.add_argument('--morphfflayerdim', type=int, default=512)
    parser.add_argument('--posrnnbidirectional', type=bool, default=True)
    parser.add_argument('--sbdrnnbidirectional', type=bool, default=True)
    parser.add_argument('--morphrnnbidirectional', type=bool, default=True)
    parser.add_argument('--posrnnnumlayers', type=int, default=1)
    parser.add_argument('--sbdrnnnumlayers', type=int, default=1)
    parser.add_argument('--morphrnnnumlayers', type=int, default=1)
    parser.add_argument('--sbdencodertype', type=str, default='lstm')
    parser.add_argument('--posencodertype', type=str, default='lstm')
    parser.add_argument('--morphencodertype', type=str, default='lstm')
    parser.add_argument('--dropout', type=float, default=0.05)
    parser.add_argument('--worddropout', type=float, default=0.05)
    parser.add_argument('--lockeddropout', type=float, default=0.5)


    args = parser.parse_args()

    if args.datatype == 'wiki':
        trainfile = '../he_iahltwiki-ud-train.conllu'
        devfile = '../he_iahltwiki-ud-dev.conllu'
    else:
        devfile = '../he_htb-ud-dev.conllu'
        trainfile = '../he_htb-ud-train.conllu'

    tagger = Tagger(trainflag=True, trainfile=trainfile, devfile=devfile, sbdrnndim=args.sbdrnndim,
                    sbdfflayerdim=args.sbdfflayerdim,
                    posrnndim=args.posrnndim, posfflayerdim=args.posfflayerdim,
                    sbdrnnbidirectional=args.sbdrnnbidirectional,
                    posrnnbidirectional=args.posrnnbidirectional, sbdrnnnumlayers=args.sbdrnnnumlayers,
                    posrnnnumlayers=args.posrnnnumlayers, sbdencodertype=args.sbdencodertype,
                    posencodertype=args.posencodertype,
                    morphrnnbidirectional=args.morphrnnbidirectional, morphrnndim=args.morphrnndim,
                    morphfflayerdim=args.morphfflayerdim, morphencodertype=args.morphencodertype,
                    morphrnnnumlayers=args.morphrnnnumlayers,
                    learningrate=args.lr, batchsize=args.trainbatch, sequencelength=args.seqlen,
                    dropout=args.dropout, wordropout=args.worddropout, lockeddropout=args.lockeddropout,
                    datatype=args.datatype)

    tagger.prepare_data_files()
    tagger.train()



if __name__ == "__main__":
    main()
