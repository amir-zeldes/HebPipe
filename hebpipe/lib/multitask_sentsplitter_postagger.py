import conllu
import torch
import torch.nn as nn
import os
import shutil


from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from lib.allennlp.conditional_random_field import ConditionalRandomField
from lib.allennlp.time_distributed import TimeDistributed



from time import time

class MTLModel(nn.Module):
    def __init__(self,rnndim=128,rnnnumlayers=1,rnnbidirectional=True,rnndropout=0.3,encodertype='lstm',ffdim=128,batchsize=32):
        super(MTLModel,self).__init__()

        self.sbdtagset = {'B-SENT':0, 'O':1}
        self.postagset = {'ADJ':0, 'ADP':1, 'ADV':2, 'AUX':3, 'CCONJ':4, 'DET':5, 'INTJ':6, 'NOUN':7, 'NUM':8, 'PRON':9, 'PROPN':10, 'PUNCT':11, 'SCONJ':12, 'SYM':13, 'VERB':14, 'X':15} # derived from HTB and IAHLTWiki trainsets #TODO: add other UD tags?

        self.sequence_length = 64
        self.batch_size = batchsize
        self.encodertype = encodertype


        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Flair embeddings do subword pooling!
        self.transformerembeddings = TransformerWordEmbeddings(model='onlplab/alephbert-base',batch_size=self.batch_size,pooling_operation='mean').to(self.device)

        # Bi-LSTM Encoder
        self.embeddingdim = 768 * 4 # based on BERT model with Flair layers
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

        # Intermediate feedforward layer
        self.ffdim = ffdim
        self.fflayer = TimeDistributed(nn.Linear(in_features=self.rnndim,out_features=self.ffdim)).to(self.device)

        # Label space for the pos tagger
        # TODO: CRF?
        self.hidden2postag = TimeDistributed(nn.Linear(in_features=self.ffdim,out_features=len(self.postagset.keys()))).to(self.device)

        # Label space for sent splitter
        self.hidden2sbd = TimeDistributed(nn.Linear(in_features=self.ffdim,out_features=len(self.sbdtagset.keys()))).to(self.device)



    def init_hidden(self):
        """
        Used by RNN-type encoders
        """
        if self.rnnbidirectional == True:
            numdir = 2
        else:
            numdir = 1

        return (torch.randn(self.rnnnumlayers * numdir, self.batch_size, self.rnndim // 2, device=self.device),
                torch.randn(self.rnnnumlayers * numdir, self.batch_size, self.rnndim // 2, device=self.device))

    def forward(self,data):

        """
        slice is a list of tuples of length = seq_len. Each tuple is (token, pos tag, sentence boundary label)
        """

        start = time()
        sents = [' '.join([s.split('\t')[0] for s in sls]) for sls in data]

        sentences = []
        for sent in sents:
            sentences.append(Sentence(sent,use_tokenizer=False))

        sentences = self.transformerembeddings.embed(sentences) # subwords are averaged here

        # Needs to extract the embeddings from Token objects, and add padding.
        embeddings = []
        for sent in sentences:
            embedding = []
            for tok in sent:
                embedding.append(tok.embedding)
            for _ in range(len(sent),self.sequence_length):
                embedding.append(torch.zeros(768 * 4)) # for padding. 4 because of the Flair TransFormerWordEmbeddings param
            embeddings.append(torch.stack(embedding))

        embeddings = torch.stack(embeddings) # final embeddings in a tensor
        print ('here timing')
        print (time() - start)

        #if self.encodertype in ('lstm','gru'):
        hidden = self.init_hidden()
        feats, hidden = self.encoder(embeddings,hidden)

        # Intermediate Feedforward layer
        feats = self.fflayer(feats)

        # logits for pos
        poslogits = self.hidden2postag(feats)
        poslogits = poslogits.permute(0,2,1)

        # logits for sbd
        sbdlogits = self.hidden2sbd(feats)

        return poslogits,sbdlogits

class Tagger():
    def __init__(self,trainflag=False,trainfile=None,devfile=None,testfile=None,rnndim=128,rnnnumlayers=1,rnnbidirectional=True,rnndropout=0.3,encodertype='lstm',ffdim=128,learningrate = 0.01):

        self.mtlmodel = MTLModel(rnndim,rnnnumlayers,rnnbidirectional,rnndropout,encodertype,ffdim)

        if trainflag == True:

            from torch.utils.tensorboard import SummaryWriter
            if os.path.isdir('../data/tensorboarddir/'):
                shutil.rmtree('../data/tensorboarddir/')
            os.mkdir('../data/tensorboarddir/')

            if not os.path.isdir('../data/checkpoint/'):
                os.mkdir('../data/checkpoint/')

            self.writer = SummaryWriter('data/tensorboarddir/')

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
        self.optimizer = torch.optim.AdamW(self.mtlmodel.parameters(), lr=learningrate)

        # Loss for pos tagging
        self.postagloss = nn.CrossEntropyLoss()
        self.postagloss.to(self.device)

        # Linear CRF Loss for sent splitter
        # for CRF
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"
        self.sbd_tag2idx = {'B-SENT': 0, 'O': 1, self.START_TAG: 2,
                            self.STOP_TAG: 3}  # AllenNLP CRF expects start and stop tags to be appended at the end, in that order
        self.sbddtransitions = [(0, 1), (1, 0), (2, 0), (2, 1), (0, 3), (1, 3)]
        self.sbdcrf = ConditionalRandomField(len(self.sbd_tag2idx) - 2, self.sbddtransitions).to(self.device)



    def train(self):

        def read_file(mode='train'):

            if mode == 'train':
                file = self.trainingdatafile
            else:
                file = self.devdatafile

            dataset = []
            with open(file,'r') as fi:
                lines = fi.readlines()
                # split into contiguous sequence of seq_len length
                for idx in range(0,len(lines),self.mtlmodel.sequence_length):
                    if idx + self.mtlmodel.sequence_length >= len(lines):
                        slice = lines[idx:len(lines)]
                    else:
                        slice = lines[idx:idx + self.mtlmodel.sequence_length]

                    dataset.append(slice)

            return dataset

        epochs = 1000

        trainingdata = read_file()
        devdata = read_file(mode='dev')

        for epoch in range(1,epochs):

            data = trainingdata[0:self.mtlmodel.batch_size]
            poslogits, sbdlogits = self.mtlmodel(data)

            sbdtags = [[s.split('\t')[2].strip() for s in sls] for sls in data]
            sbdtags = [[self.sbd_tag2idx[t] for t in tag] for tag in sbdtags]
            sbdtags = torch.LongTensor(sbdtags).to(self.device)

            postags = [[s.split('\t')[1].strip() for s in sls] for sls in data]
            postags = [[self.mtlmodel.postagset[t] for t in tag] for tag in postags]
            postags = torch.LongTensor(postags).to(self.device)

            posloss = self.postagloss(poslogits,postags)
            sbdloss = self.sbdcrf(sbdlogits,sbdtags) * -1

            mtlloss = posloss + sbdloss # uniform weighting. # TODO: learnable weights?

            self.optimizer.zero_grad()
            mtlloss.backward()
            self.optimizer.step()

            self.writer.add_scalar('train_pos_loss', posloss.item(), epoch)
            self.writer.add_scalar('train_sbd_loss', sbdloss.item(), epoch)
            self.writer.add_scalar('train_joint_loss', mtlloss.item(), epoch)


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
    #tagger.prepare_data_files()
    tagger.train()

    print ('here')


if __name__ == "__main__":
    main()