#!/usr/bin/python
# -*- coding: utf-8 -*-


def toks_to_sents(text, split_at_punct=True):

	lines = text.split("\n")
	output = []
	if not lines[0] == "<s>":
		output.append("<s>")

	for i, line in enumerate(lines):
		if len(output) == 0:
			output.append(line)
		else:
			if not (line == "<s>" and output[-1] == "<s>") and not (line == "</s>" and output[-1] == "<s>"):
				output.append(line)
		if line in [".","!","?"] and split_at_punct:
			output.append("</s>")
			output.append("<s>")

	if output[-1] != "</s>":
		output.append("</s>")

	return "\n".join(output)
