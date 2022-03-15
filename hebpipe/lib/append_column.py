#!/usr/bin/python
# -*- coding: utf-8 -*-

import io, sys
from argparse import ArgumentParser

#this script takes a text file with tab delimited columns
#and adds every column except the first line-wise to the target file.
#Both files should ideally have the same number of lines.
#The script will terminate once either file ends.
# For example:
# Appended file:
# Computer	Nom.Sg.Masc
#
# Target file:
# Computer	NN	Computer 
#
# becomes
#
# Computer	NN	Computer	Nom.Sg.Masc
#
# usage:
# append_column.py [OPTIONS] <APPEND_COLS_FILE> <TARGET_FILE>
#
# Options and arguments:
#
# -h           Print this message and quit
#
#
# example:
# append_column.py morph.tab tagged.tab > merged.tab
# etc.

PY3 = sys.version_info[0] == 3


def inject_col(source_lines, target_lines, col=-1, into_col=None, skip_supertoks=False):

	output = []
	counter = -1
	target_line = ""

	if not PY3:
		if isinstance(target_lines,unicode):
			target_lines = str(target_lines.encode("utf8"))

	if not isinstance(source_lines,list):
		source_lines = source_lines.split("\n")
	if not isinstance(target_lines,list):
		target_lines = target_lines.split("\n")

	if col != -1:
		# non-final column requested, ensure source_lines only has lines with tabs
		source_lines = [l for l in source_lines if "\t" in l]

	for i, source_line in enumerate(source_lines):
		while len(target_line) == 0:
			counter +=1
			target_line = target_lines[counter]
			if (target_line.startswith("<") and target_line.endswith(">")) or len(target_line) == 0:
				output.append(target_line)
				target_line = ""
			else:
				target_cols = target_line.split("\t")
				if "-" in target_cols[0] and skip_supertoks:
					output.append(target_line)
					target_line = ""
		source_cols = source_line.split("\t")
		to_inject = source_cols[col]
		target_cols = target_line.split("\t")
		if into_col is None:
			target_cols.append(to_inject)
		else:
			target_cols[into_col] = to_inject
		output.append("\t".join(target_cols))
		target_line=""

	return "\n".join(output)


if __name__ == "__main__":
	p = ArgumentParser()
	p.add_argument("source")
	p.add_argument("target")
	p.add_argument("-c","--col",action="store",default="-1")

	opts = p.parse_args()
	source_lines = io.open(opts.file1,encoding="utf8").read().replace("\r","").split("\n")
	target_lines = io.open(opts.file1,encoding="utf8").read().replace("\r","").split("\n")
	merged = inject_col(source_lines,target_lines,opts.col)
	print(merged)

