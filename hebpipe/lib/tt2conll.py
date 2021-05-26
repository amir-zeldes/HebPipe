#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
# Python port of TT2CoNLL.pl
#
# this script takes a corpus in CWB SGML format
# and converts it into the CoNLL input format using
# the POS tag column (column 2) as both the fine
# and coarse grained tag. SGML elements are deleted.
#
# Computer	NN	Computer
#
# becomes
#
# 1	Computer	Computer	NN	NN	_	_	_
#
# The script splits sentences based on a predetermined sentence
# splitting tag, at the moment defaulting to $. (STTS sentence
# punctuation)
#
# usage:
# python tt2conll.py [OPTIONS] <CORPUS>
#
# Options and arguments:
#
# -t <tagname> Use the tag <tagname> to split the corpus into sentences, default is $.
# -x <element> Use the XML element <element> to split the corpus into sentences (e.g. s)
# -h           Print this message and quit
#
# <CORPUS>     A corpus coded in CWB SGML
#
# examples:
# python tt2conll.py -t $. my_corpus.sgml > my_corpus.conll
# python tt2conll.py -t SENT my_corpus.sgml > my_corpus.conll
# python tt2conll.py -x s my_corpus.sgml > my_corpus.conll
# etc.
"""

from argparse import ArgumentParser
import io, sys, re

PY3 = sys.version_info[0] == 3


def conllize(in_text,tag=None,element=None,no_zero=False, super_mapping=None, ten_cols=False, attrs_as_comments=False):

	if not PY3:
		if not isinstance(in_text,unicode):
			in_text = unicode(in_text.decode("utf8"))

	xml = False
	if element is not None:
		xml = True

	outlines = []
	counter = 1
	for line in in_text.replace("\r","").split("\n"):
		if len(line.strip()) > 0:
			tabs = line.count("\t")
			lemma = "_"
			pos = "_"
			morph = "_"
			misc = "_"
			if tabs == 0:
				tok = line
			else:
				fields = line.split("\t")
				tok = fields[0]
				pos = fields[1]
				if tabs > 1:
					lemma = fields[2]
				if tabs > 2:
					morph = fields[3]
				if tabs > 3:
					misc = fields[4]
			if not (line.startswith("<") and line.endswith(">")):  # Do not make tokens out of XML elements
				if not ten_cols:
					if no_zero:
						outlines.append("\t".join([str(counter), tok, lemma, pos, pos, morph, "_", "_"]))
					else:
						outlines.append("\t".join([str(counter),tok,lemma,pos,pos,morph,"0","_"]))
				else:
					if no_zero:
						outlines.append("\t".join([str(counter), tok, lemma, pos, pos, morph, "_", "_", "_", misc]))
					else:
						outlines.append("\t".join([str(counter), tok, lemma, pos, pos, morph, "0", "_", "_", misc]))
				counter += 1
			if xml:
				if "<" + element + " " in line and attrs_as_comments:
					attrs = re.findall(r' ([^ ="]+)="([^"]+)"',line)
					for key, val in attrs:
						outlines.append("# " + key + " = " + val.strip())
				if "</" + element + ">" in line:
					counter = 1
					outlines.append("")
			else:
				if pos == tag:
					counter = 1
					outlines.append("")
	if super_mapping is None:
		return "\n".join(outlines)
	else:
		return add_supertokens(outlines, super_mapping)


def add_supertokens(conll_lines,mapping):

	output = []
	counter = 0

	for line in conll_lines:
		if "\t" in line and not line.startswith("#"):
			tok_id = line.split("\t")[0]
			if "-" in tok_id:
				output.append(line)
				continue
			if counter in mapping:
				super_tok, super_len = mapping[counter]
				super_id = tok_id + "-" + str(int(tok_id)+super_len-1)
				super_line = "\t".join([super_id,super_tok] + ["_"]*8)
				output.append(super_line)
			counter+=1

		output.append(line)

	return "\n".join(output)


if __name__ == "__main__":
	parser = ArgumentParser()
	parser.add_argument("-t","--tagname",action="store",default="$.",help="Use the POS tag <tagname> to split the corpus into sentences, default is $.")
	parser.add_argument("-x","--xml_element",action="store",default=None,help="Use the XML element <element> to split the corpus into sentences (e.g. s)")
	parser.add_argument("infile",action="store",help="file to process")

	opts = parser.parse_args()

	in_data = io.open(opts.infile,encoding="utf8").read().replace("\r","")
	conllized = conllize(in_data,opts.tagname,opts.xml_element)
	print(conllized)

