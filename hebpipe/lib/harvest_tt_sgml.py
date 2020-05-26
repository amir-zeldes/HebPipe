"""
Script to read Scriptorium style SGML
into plain TT SGML: norm -> tok etc.
"""

import sys, io, re


def harvest_tt(text, keep_sgml=True):

	norm, pos, lemma = "","",""
	output = ""

	for line in text.split("\n"):
		m = re.search(' norm="([^"]+)"',line)
		if m is not None:
			norm = m.group(1)
		m = re.search(' lemma="([^"]+)"',line)
		if m is not None:
			lemma = m.group(1)
		m = re.search(' pos="([^"]+)"',line)
		if m is not None:
			pos = m.group(1)

		if keep_sgml:
			if "<" in line and re.search(r'(norm|pos|lemma)=|</(norm|pos|lemma)>',line) is None:
				output += line + "\n"

		if "</norm>" in line:
			if norm !="" and norm != " ":
				output += "\t".join([norm,pos,lemma])+"\n"

	return output


if __name__ == "__main__":

	if sys.argv[1] == "-g":
		from glob import glob
		files = glob(sys.argv[2])
		all_out = ""
		for file_ in files:
			input_text = io.open(file_, encoding="utf8").read().replace("\r", "")
			out_ = harvest_tt(input_text,keep_sgml=False)
			all_out += out_
		sys.stdout.buffer.write((all_out + "\n").encode('utf8'))

	else:

		input_text = io.open(sys.argv[1],encoding="utf8").read().replace("\r","")

		out_ = harvest_tt(input_text)
		sys.stdout.buffer.write((out_ + "\n").encode('utf8'))

