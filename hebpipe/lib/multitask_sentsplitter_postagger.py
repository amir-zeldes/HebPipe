import conllu
import torch
import torch.nn as nn

from flair.embeddings import TransformerWordEmbeddings
from flair.data import Sentence
from lib.allennlp.conditional_random_field import ConditionalRandomField
from lib.allennlp.time_distributed import TimeDistributed


class MTLModel(nn.Module):
    def __init__(self,rnndim=128,rnnnumlayers=1,rnnbidirectional=True,rnndropout=0.3,encodertype='lstm',ffdim=128):
        super(MTLModel,self).__init__()

        self.sbdtagset = ['B-SENT', 'O']
        self.postagset = ['ADJ', 'ADP', 'ADV', 'AUX', 'CCONJ', 'DET', 'INTJ', 'NOUN', 'NUM', 'PRON', 'PROPN', 'PUNCT', 'SCONJ', 'SYM', 'VERB', 'X'] # derived from HTB and IAHLTWiki trainsets #TODO: add other UD tags?

        self.sequence_length = 256
        self.batch_size = 5

        # for CRF
        self.START_TAG = "<START>"
        self.STOP_TAG = "<STOP>"

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Flair embeddings do subword pooling!
        self.transformerembeddings = TransformerWordEmbeddings(model='onlplab/alephbert-base',batch_size=self.batch_size,pooling_operation='mean').to(self.device)

        # Bi-LSTM Encoder
        self.embeddingdim = 768 # based on BERT model
        self.rnndim = rnndim
        self.rnnnumlayers = rnnnumlayers
        self.rnnbidirectional = rnnbidirectional
        self.rnndropout = rnndropout


        if encodertype == 'lstm':
            self.encoder = nn.LSTM(input_size=self.embeddingdim, hidden_size=self.rnndim // 2,
                                 num_layers=self.rnnnumlayers, bidirectional=self.rnnbidirectional,
                                 dropout=self.rnndropout).to(self.device)
        elif encodertype == 'gru':
            self.encoder = nn.GRU(input_size=self.embeddingdim, hidden_size=self.rnndim // 2,
                                   num_layers=self.rnnnumlayers, bidirectional=self.rnnbidirectional,
                                   dropout=self.rnndropout).to(self.device)

        # Intermediate feedforward layer
        self.ffdim = ffdim
        self.fflayer = TimeDistributed(nn.Linear(in_features=self.rnndim,out_features=self.ffdim)).to(self.device)

        # Label space for the pos tagger
        # TODO: CRF?
        self.hidden2postag = TimeDistributed(nn.Linear(in_features=self.ffdim,out_features=len(self.postagset))).to(self.device)

        # Label space for sent splitter
        self.hidden2sbd = TimeDistributed(nn.Linear(in_features=self.ffdim,out_features=len(self.sbdtagset))).to(self.device)

        # Linear CRF for sent splitter
        self.sbd_tag2idx = {'B-SENT':0,'O':1,self.START_TAG:2,self.STOP_TAG:3} # AllenNLP CRF expects start and stop tags to be appended at the end, in that order
        self.sbddtransitions = [(0,1),(1,0),(2,0),(2,1),(0,3),(1,3)]
        self.sbdcrf = ConditionalRandomField(len(self.sbd_tag2idx),self.sbddtransitions).to(self.device)

    def init_hidden(self):
        """
        Used by RNN-type encoders
        """
        if self.rnnbidirectional == True:
            numdir = 2
        else:
            numdir = 1

        return (torch.randn(self.rnnnumlayers * numdir, 1, self.rnndim // 2, device=self.device),
                torch.randn(self.rnnnumlayers * numdir, 1, self.rnndim // 2, device=self.device))

    def forward(self,slice):

        """
        slice is a list of tuples of length = seq_len. Each tuple is (token, pos tag, sentence boundary label)
        """

        sents = [' '.join([s.split('\t')[0] for s in sls]) for sls in slice]

        sentences = []
        for sent in sents:
            sentences.append(Sentence(sent,use_tokenizer=False))

        sentences = self.transformerembeddings.embed(sentences)

        embeddings = []
        for sent in sentences:
            embedding = []
            for tok in sent:
                embedding.append(tok.embedding)
            for _ in range(len(sent),self.sequence_length):
                embedding.append(torch.zeros(768 * 4)) # for padding
            embeddings.append(torch.stack(embedding))

        embeddings = torch.stack(embeddings)
        print ('here')


class Tagger():
    def __init__(self,trainflag=False,trainfile=None,devfile=None,testfile=None,rnndim=128,rnnnumlayers=1,rnnbidirectional=True,rnndropout=0.3,encodertype='lstm',ffdim=128):

        self.mtlmodel = MTLModel(rnndim,rnnnumlayers,rnnbidirectional,rnndropout,encodertype,ffdim)

        if trainflag == True:
            import tensorboard
            self.trainingdatafile = '../data/sentsplit_postag_train_gold.tab'
            self.devdatafile = '../data/sentsplit_postag_dev_gold.tab'
        else:
            self.testdatafile = '../data/sentsplit_postag_test_gold.tab'


        self.trainflag = trainflag
        self.trainfile = trainfile
        self.devfile = devfile
        self.testfile = testfile


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

        trainingdata = read_file()
        devdata = read_file(mode='dev')


        self.mtlmodel(trainingdata[0:5])



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