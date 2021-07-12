"""
flair_pos_tagger.py

This module trains flair sequence labelers to predict POS and deprel for OTHER modules.
"""


from argparse import ArgumentParser
from flair.data import Corpus, Sentence
from flair.datasets import ColumnCorpus
from flair.embeddings import OneHotEmbeddings, TransformerWordEmbeddings, StackedEmbeddings
from flair.models import SequenceTagger
import os, sys, io
from glob import glob

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep
model_dir = script_dir + ".." + os.sep + "models" + os.sep
IAHLT_ROOT = "C:\\Uni\\Corpora\\Hebrew\\IAHLT_HTB\\"  # Path to IAHLT HTB repo

class FlairTagger:

    def __init__(self, train=False):
        if not train:
            model_name = model_dir + "heb.flair"
            self.model = SequenceTagger.load(model_name)

    @staticmethod
    def make_pos_data():
        files = glob(IAHLT_ROOT + "*.conllu")
        train = test = dev = ""
        super_tok_len = 0
        super_tok_start = False
        for file_ in files:
            output = []
            lines = io.open(file_,encoding="utf8").readlines()
            for line in lines:
                if "\t" in line:
                    fields = line.split("\t")
                    if "." in fields[0]:
                        continue
                    if "-" in fields[0]:
                        super_tok_start = True
                        start,end = fields[0].split("-")
                        super_tok_len = int(end)-int(start) + 1
                        continue
                    if super_tok_start:
                        super_tok_position = "B"
                        super_tok_start = False
                        super_tok_len -= 1
                    elif super_tok_len > 0:
                        super_tok_position = "I"
                        super_tok_len -= 1
                        if super_tok_len == 0:
                            super_tok_position = "E"
                    else:
                        super_tok_position = "O"
                    output.append(fields[1] + "\t" + super_tok_position + "\t" + fields[4])
                elif len(line.strip()) == 0:
                    if output[-1] != "":
                        output.append("")
            if "dev" in file_:
                dev += "\n".join(output)
            elif "test" in file_:
                test += "\n".join(output)
            else:
                train += "\n".join(output)
        with io.open("tagger" + os.sep + "heb_train.txt", 'w', encoding="utf8",newline="\n") as f:
            f.write(train)
        with io.open("tagger" + os.sep + "heb_dev.txt", 'w', encoding="utf8",newline="\n") as f:
            f.write(dev)
        with io.open("tagger" + os.sep + "heb_test.txt", 'w', encoding="utf8",newline="\n") as f:
            f.write(test)

    def train(self, cuda_safe=True, positional=True):
        if cuda_safe:
            # Prevent CUDA Launch Failure random error, but slower:
            import torch
            torch.backends.cudnn.enabled = False
            # Or:
            # os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

        # 1. get the corpus
        # this is the folder in which train, test and dev files reside
        data_folder = "tagger" + os.sep

        # init a corpus using column format, data folder and the names of the train, dev and test files

        # define columns
        columns = {0: "text", 1: "super", 2: "pos"}

        self.make_pos_data()

        corpus: Corpus = ColumnCorpus(
            data_folder, columns,
            train_file="heb_train.txt",
            test_file="heb_test.txt",
            dev_file="heb_dev.txt",
        )

        # 2. what tag do we want to predict?
        tag_type = 'pos'

        # 3. make the tag dictionary from the corpus
        tag_dictionary = corpus.make_tag_dictionary(tag_type=tag_type)
        print(tag_dictionary)

        # 4. initialize embeddings
        embeddings: TransformerWordEmbeddings = TransformerWordEmbeddings('onlplab/alephbert-base')
        if positional:
            positions: OneHotEmbeddings = OneHotEmbeddings(corpus=corpus, field="super")
            stacked: StackedEmbeddings = StackedEmbeddings([embeddings,positions])
        else:
            stacked = embeddings

        # 5. initialize sequence tagger
        tagger: SequenceTagger = SequenceTagger(hidden_size=256,
                                                embeddings=stacked,
                                                tag_dictionary=tag_dictionary,
                                                tag_type=tag_type,
                                                use_crf=True,
                                                use_rnn=True)

        # 6. initialize trainer
        from flair.trainers import ModelTrainer

        trainer: ModelTrainer = ModelTrainer(tagger, corpus)

        # 7. start training
        trainer.train(script_dir + "pos-dependencies" + os.sep + 'flair_tagger',
                      learning_rate=0.1,
                      mini_batch_size=15,
                      max_epochs=150)

    def predict(self, in_path=None, in_format="flair", out_format="conllu", as_text=False):
        model = self.model
        tagcol = 4

        if as_text:
            data = in_path
            #data = (data + "\n").replace("<s>\n", "").replace("</s>\n", "\n").strip()
        else:
            data = io.open(in_path,encoding="utf8").read()
        sents = []
        words = []
        positions = []
        true_tags = []
        true_pos = []
        super_tok_start = False
        super_tok_len = 0
        data = data.strip() + "\n"  # Ensure final new line for last sentence
        for line in data.split("\n"):
            if len(line.strip())==0:
                if len(words) > 0:
                    sents.append(Sentence(" ".join(words),use_tokenizer=lambda x:x.split(" ")))
                    for i, word in enumerate(sents[-1]):
                        word.add_label("super",positions[i])
                    words = []
                    positions = []
            else:
                if in_format == "flair":
                    words.append(line.split("\t")[0])
                    positions.append(line.split("\t")[1])
                    true_tags.append(line.split("\t")[2]) if "\t" in line else true_tags.append("")
                else:
                    if "\t" in line:
                        fields = line.split("\t")
                        if "." in fields[0]:
                            continue
                        if "-" in fields[0]:
                            super_tok_start = True
                            start, end = fields[0].split("-")
                            super_tok_len = int(end) - int(start) + 1
                            continue
                        if super_tok_start:
                            super_tok_position = "B"
                            super_tok_start = False
                            super_tok_len -= 1
                        elif super_tok_len > 0:
                            super_tok_position = "I"
                            super_tok_len -= 1
                            if super_tok_len == 0:
                                super_tok_position = "E"
                        else:
                            super_tok_position = "O"
                        words.append(line.split("\t")[1])
                        positions.append(super_tok_position)
                        true_tags.append(line.split("\t")[tagcol])
                        true_pos.append(line.split("\t")[4])

        # predict tags and print
        model.predict(sents)#, all_tag_prob=True)

        preds = []
        scores = []
        words = []
        for i, sent in enumerate(sents):
            for tok in sent.tokens:
                pred = tok.labels[1].value
                score = str(tok.labels[1].score)
                preds.append(pred)
                scores.append(score)
                words.append(tok.text)

        do_postprocess = False
        if do_postprocess:
            preds, scores = self.post_process(words, preds, scores)

        toknum = 0
        output = []
        #out_format="diff"
        for i, sent in enumerate(sents):
            tid=1
            if i>0 and out_format=="conllu":
                output.append("")
            for tok in sent.tokens:
                pred = preds[toknum]
                score = str(scores[toknum])
                if len(score)>5:
                    score = score[:5]
                if out_format == "conllu":
                    fields = [str(tid),tok.text,"_",pred,pred,"_","_","_","_"]
                    output.append("\t".join(fields))
                    tid+=1
                elif out_format == "xg":
                    output.append("\t".join([pred, tok.text, score]))
                else:
                    true_tag = true_tags[toknum]
                    corr = "T" if true_tag == pred else "F"
                    output.append("\t".join([pred, true_tag, corr, score, tok.text, true_pos[toknum]]))
                toknum += 1

        if as_text:
            return "\n".join(output)
        else:
            ext = "xpos.conllu" if out_format == "conllu" else "txt"
            partition = "test" if "test" in in_path else "dev"
            with io.open(script_dir + "pos-dependencies" +os.sep + "flair-"+partition+"-pred." + ext,'w',encoding="utf8",newline="\n") as f:
                f.write("\n".join(output))

    @staticmethod
    def post_process(word_list, pred_list, score_list, softmax_list=None):
        """
        Implement a subset of closed-class words that can only take one of their attested closed class POS tags
        """
        output = []

        KNOWN_PUNCT = {'’', '“', '”'}
        closed = {"except":["IN"],
                  "or":["CC"],
                  "another":["DT"],
                  "be":["VB"]
                  }
        # case marking VVG can never be IN:
        vbg_preps = {("including","IN"):"VBG",("according","IN"):"VBG",("depending","IN"):"VBG",("following","IN"):"VBG",("involving","IN"):"VBG",
                     ("regarding","IN"):"VBG",("concerning","IN"):"VBG"}

        top100 = {",":",",".":".","of":"IN","is":"VBZ","you":"PRP","for":"IN","was":"VBD","with":"IN","The":"DT","are":"VBP",")":"-RRB-","(":"-LRB-","at":"IN","this":"DT","from":"IN","or":"CC","not":"RB","his":"PRP$","they":"PRP","an":"DT","we":"PRP","n't":"RB","he":"PRP","[":"-LRB-","]":"-RRB-","has":"VBZ","my":"PRP$","their":"PRP$","It":"PRP","were":"VBD","In":"IN","if":"IN","would":"MD","”":"''",";":":","into":"IN","when":"WRB","You":"PRP","also":"RB","she":"PRP","our":"PRP$","been":"VBN","who":"WP","We":"PRP","time":"NN","He":"PRP","This":"DT","its":"PRP$","did":"VBD","two":"CD","these":"DT","many":"JJ","And":"CC","!":".","should":"MD","because":"IN","how":"WRB","If":"IN","n’t":"RB","'re":"VBP","him":"PRP","'m":"VBP","city":"NN","could":"MD","may":"MD","years":"NNS","She":"PRP","really":"RB","now":"RB","new":"JJ","something":"NN","here":"RB","world":"NN","They":"PRP","life":"NN","But":"CC","year":"NN","us":"PRP","between":"IN","different":"JJ","those":"DT","language":"NN","does":"VBZ","same":"JJ","going":"VBG","United":"NNP","day":"NN","few":"JJ","For":"IN","every":"DT","important":"JJ","When":"WRB","things":"NNS","during":"IN","might":"MD","kind":"NN","How":"WRB","system":"NN","thing":"NN","example":"NN","another":"DT","small":"JJ","until":"IN","information":"NN","away":"RB"}

        scores = []

        #VBG must end in ing/in; VBN may not
        for i, word in enumerate(word_list):
            pred = pred_list[i]
            score = score_list[i]
            if word in top100:
                output.append(top100[word])
                scores.append("_")
            elif (word.lower(),pred) in vbg_preps:
                output.append(vbg_preps[(word.lower(),pred)])
                scores.append("_")
            else:
                output.append(pred)
                scores.append(score)

        return output, scores


if __name__ == "__main__":
    p = ArgumentParser()
    p.add_argument("-m","--mode",choices=["train","predict"],default="predict")
    p.add_argument("-f","--file",default=None,help="Blank for training, blank predict for eval, or file to run predict on")
    p.add_argument("-p","--positional_embeddings",help="Whether to use positional embeddings within supertokens (MWTs)")
    p.add_argument("-i","--input_format",choices=["flair","conllu"],default="flair",help="flair two column training format or conllu")
    p.add_argument("-o","--output_format",choices=["flair","conllu","xg"],default="conllu",help="flair two column training format or conllu")

    opts = p.parse_args()

    if opts.mode == "train":
        tagger = FlairTagger(train=True)
        tagger.train(positional=opts.positional_embeddings)
    else:
        tagger = FlairTagger(train=False)
        tagger.predict(in_format=opts.input_format, out_format=opts.output_format,
                in_path=opts.file)

#-m predict -i conllu -f C:\Uni\Corpora\Hebrew\IAHLT_HTB\he_htb-ud-test.conllu