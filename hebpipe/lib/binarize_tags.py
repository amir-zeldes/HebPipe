"""
Python port of Perl script binarize_tags.pl

Binarizes unary XML milestone elements by assuming each milestone element begin a span stretching until the
next occurrence of the same element, or the end of the file
"""

from argparse import ArgumentParser
import io, re
from collections import defaultdict

def binarize(in_data):
	rev_lines = in_data.split("\n")[::-1]
	attrs = {}
	unary = defaultdict(int)
	edited_lines = []
	for line in rev_lines:
		elements = re.findall(r'<([^> ]+)([^>]*)/>',line)
		for tag, ats in elements:
			attrs[tag] = ats
			unary[tag] += 1
			line = re.sub(r'<' + tag + '[^>]*/>',r'</'+tag+'><'+tag+attrs[tag]+">",line)
		edited_lines.append(line)

	count = defaultdict(int)
	outlines = []
	for line in edited_lines[::-1]:
		tags = re.findall(r'</([^> ]+)>',line)
		for tag in tags:
			if tag in unary:
				count[tag]+=1

			if count[tag] == 1 and tag in unary:  # Destroy superfluous first closing tag
				line = line.replace("</"+tag+">","",1)
		outlines.append(line)

	finals = []
	for tag in sorted(unary.keys()):  # Final closing tags
		finals.append("</" + tag + ">")

	return "\n".join(outlines) + "\n".join(finals) + "\n"


if __name__=="__main__":
	parser = ArgumentParser()
	parser.add_argument("infile",action="store",help="file to process")

	opts = parser.parse_args()

	in_data = io.open(opts.infile,encoding="utf8").read().replace("\r","")
	binarized = binarize(in_data)
	print(binarized)

