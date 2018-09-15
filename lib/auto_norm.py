#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
auto_norm - Python
V2.0.0
Python port of the Perl script normalizing Sahidic Coptic text to standard spelling.

Usage:  auto_norm.py [options] <FILE>

Options and argument:

-h              print this [h]elp message and quit
-s              use [s]ahidica Bible specific normalization rules
-t              use [t]able containing previous normalizations (first column is diplomatic text, last column is normalized)

<FILE>    A text file encoded in UTF-8 without BOM


Examples:

Normalize a Coptic plain text file in UTF-8 encoding (without BOM):
python auto_norm.py in_Coptic_utf8.txt > out_Coptic_normalized.txt
python auto_norm.py -t norm_table.tab in_Coptic_utf8.txt > out_Coptic_normalized.txt

Copyright 2013-2018, Amir Zeldes & Caroline T. Schroeder

This program is free software.
"""

from argparse import ArgumentParser
import io, sys, os, re


def normalize(in_data,table_file=None,sahidica=False):

	outlines=[]
	if table_file is None:
		# Use default location for norm table
		table_file = "data" + os.sep + "norm_table.tab"
	try:
		norm_lines = io.open(table_file,encoding="utf8").read().replace("\r","").split("\n")
	except IOError as e:
		sys.stderr.write("could not find normalization table file at " + table_file + "\n")
		sys.exit(0)
	norms = dict((line.split("\t")) for line in norm_lines if "\t" in line)
	for line in in_data.split("\n"):
		if line in norms:
			line = norms[line]
		else:
			line = line.replace("|","").replace("[","").replace("]","")
			line = line.replace("⳯","ⲛ")
			line = line.replace("[`̂︦︥̄⳿̣̣̇̈̇̄̈︤᷍]","")
			line = line.replace("Ⲁ","ⲁ")
			line = line.replace("Ⲃ","ⲃ")
			line = line.replace("Ⲅ","ⲅ")
			line = line.replace("Ⲇ","ⲇ")
			line = line.replace("Ⲉ","ⲉ")
			line = line.replace("Ϥ","ϥ")
			line = line.replace("Ⲫ","ⲫ")
			line = line.replace("Ⲍ","ⲍ")
			line = line.replace("Ⲏ","ⲏ")
			line = line.replace("Ⲑ","ⲑ")
			line = line.replace("Ⲓ","ⲓ")
			line = line.replace("Ⲕ","ⲕ")
			line = line.replace("Ⲗ","ⲗ")
			line = line.replace("Ⲙ","ⲙ")
			line = line.replace("Ⲛ","ⲛ")
			line = line.replace("Ⲟ","ⲟ")
			line = line.replace("Ⲝ","ⲝ")
			line = line.replace("Ⲡ","ⲡ")
			line = line.replace("Ⲣ","ⲣ")
			line = line.replace("Ⲥ","ⲥ")
			line = line.replace("Ⲧ","ⲧ")
			line = line.replace("Ⲩ","ⲩ")
			line = line.replace("Ⲱ","ⲱ")
			line = line.replace("Ⲯ","ⲯ")
			line = line.replace("Ⲭ","ⲭ")
			line = line.replace("Ϩ","ϩ")
			line = line.replace("Ϫ","ϫ")
			line = line.replace("Ϣ","ϣ")
			line = line.replace("Ϭ","ϭ")
			line = line.replace("Ϯ","ϯ")
			line = line.replace("̂","")
			line = line.replace("`","")
			line = line.replace("᷍","")
			line = line.replace("̣","")

			line = re.sub(r"(^|_)ⲓⲏⲗ(|_)", r"\1ⲓⲥⲣⲁⲏⲗ\2", line)
			line = re.sub(r"(^|_)ⲓⲏ?ⲥ(|_)", r"\1ⲓⲏⲥⲟⲩⲥ\2", line)
			line = re.sub(r"(^|_)ϫⲟⲓⲥ(|_)", r"\1ϫⲟⲉⲓⲥ\2", line)
			line = re.sub(r"(^|_)ⲭⲣ?ⲥ(|_)", r"\1ⲭⲣⲓⲥⲧⲟⲥ\2", line)
			line = re.sub(r"(^|_)ϯⲟⲩⲇⲁⲓⲁ(|_)", r"\1ⲧⲓⲟⲩⲇⲁⲓⲁ\2", line)
			line = re.sub(r"(^|_)ⲡⲛⲁ(|_)", r"\1ⲡⲛⲉⲩⲙⲁ\2", line)
			line = re.sub(r"(^|_)ⲃⲁⲍⲁⲛⲓⲍⲉ(|_)", r"\1ⲃⲁⲥⲁⲛⲓⲍⲉ\2", line)
			line = re.sub(r"(^|_)ⲃⲁⲍⲁⲛⲟⲥ(|_)", r"\1ⲃⲁⲥⲁⲛⲟⲥ\2", line)
			line = re.sub(r"(^|_)ϩⲓⲗⲏⲙ(|_)", r"\1ϩⲓⲉⲣⲟⲩⲥⲁⲗⲏⲙ\2", line)
			line = re.sub(r"(^|_)ⲥ[ⳁⲣ]ⲟⲥ(|_)", r"\1ⲥⲧⲁⲩⲣⲟⲥ\2", line)
			line = re.sub(r"(^|_)ⲕⲗⲏⲣⲟⲛⲟⲙⲓ(|_)", r"\1ⲕⲗⲏⲣⲟⲛⲟⲙⲉⲓ\2", line)
			line = re.sub(r"(^|_)ⲓⲱⲧ(|_)", r"\1ⲉⲓⲱⲧ\2", line)
			line = re.sub(r"(^|_)ⲓⲟⲧⲉ(|_)", r"\1ⲉⲓⲟⲧⲉ\2", line)
			line = re.sub(r"(^|_)ϩⲣⲁⲉⲓ(|_)", r"\1ϩⲣⲁⲓ\2", line)
			line = re.sub(r"(^|_)ⲡⲏⲟⲩⲉ(|_)", r"\1ⲡⲏⲩⲉ\2", line)
			line = re.sub(r"(^|_)ϩⲃⲏⲟⲩⲉ(|_)", r"\1ϩⲃⲏⲩⲉ\2", line)
			line = re.sub(r"(^|_)ⲓⲉⲣⲟⲥⲟⲗⲩⲙⲁ(|_)", r"\1ϩⲓⲉⲣⲟⲩⲥⲁⲗⲏⲙ\2", line)
			line = re.sub(r"(^|_)ⲡⲓⲑⲉ(|_)", r"\1ⲡⲉⲓⲑⲉ\2", line)
			line = re.sub(r"(^|_)ⲡⲣⲟⲥⲕⲁⲣⲧⲉⲣⲓ(|_)", r"\1ⲡⲣⲟⲥⲕⲁⲣⲧⲉⲣⲓⲁ\2", line)
			line = re.sub(r"(^|_)ⲙⲡⲁⲧⲉ[ⲕϥⲥⲛ]([^_ ]*)(|_)", r"\1ⲙⲡⲁⲧ\2\3", line)

			# Sahidica specific replacements
			if sahidica:
				line = re.sub(r'ⲟⲉⲓ($|_| )',"ⲉⲓ",line)
				line = re.sub(r'^([ⲡⲧⲛ])ⲉⲉⲓ)',r"\1ⲉⲓ",line)

		outlines.append(line)
	return "\n".join(outlines)


if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument("-s","--sahidica",action="store_true",help="use [s]ahidica Bible specific normalization rules")
	parser.add_argument("-t","--table",action="store",default=None,help="use [t]able containing previous normalizations (first column is diplomatic text, last column is normalized)")
	parser.add_argument("infile",action="store",help="file to process")

	opts = parser.parse_args()

	in_data = io.open(opts.infile,encoding="utf8").read().replace("\r","")
	normalized = normalize(in_data,table_file=opts.table,sahidica=opts.sahidica)
	print(normalized)

