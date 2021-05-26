"""
build_sent_data.py

Creates train/dev/test data in flair format to train flair_sent_splitter

Assumes three files in the current directory, e.g. dev.conllu, test.conllu, train.conllu
Files for flair training will be created in the current directory and should be *MOVED*
to lib/flair/data/ before invoking training in flair_sent_splitter.py
"""

from glob import glob
import os, sys, re, io
from collections import defaultdict

conll_dir = "" + os.sep  # CoNLL-U training data is assumed to be in lib/flair/*.conllu, or set a different path here

files = glob(conll_dir + "*.conllu")

data = defaultdict(list)

for file_ in files:
    doc = os.path.basename(file_).replace(".conllu", "")

    partition = "train"
    if "test" in file_:
        partition = "test"
    elif "dev" in file_:
        partition = "dev"

    lines = io.open(file_, encoding="utf8").read().strip().split("\n")

    sent = "B-SENT"
    counter = 0
    for line in lines:
        if len(line.strip()) == 0:
            sent = "B-SENT"
        if "\t" in line:
            fields = line.split("\t")
            if "-" in fields[0]:
                continue
            word = fields[1]
            pos = fields[4]
            data[partition].append(word + " " + sent)
            sent = "O"
            counter += 1
            if counter == 21:
                #data[partition].append("")
                #data[partition].append("-DOCSTART- X")
                data[partition].append("")
                counter = 0

    data[partition].append("")

for partition in data:
    lines = data[partition]
    with io.open("sent_" + partition + ".txt", "w", encoding="utf8", newline="\n") as f:
        f.write("\n".join(lines) + "\n")
