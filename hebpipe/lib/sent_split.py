#!/usr/bin/python
# -*- coding: utf-8 -*-


def toks_to_sents(lines, split_at_punct=True, as_lists=True):

	if not isinstance(lines,list):
		lines = lines.split("\n")
	output = []
	if not lines[0] == "<s>" and not as_lists:
		output.append("<s>")

	for i, line in enumerate(lines):
		if line == "<s>" and as_lists:
			continue
		if len(output) == 0:
			output.append(line)
		else:
			if not (line == "<s>" and output[-1] == "<s>") and not (line == "</s>" and output[-1] == "<s>"):
				output.append(line)
		if line in [".","!","?","...","…","!?","?!"] and split_at_punct:
			output.append("</s>")
			if not as_lists:
				output.append("<s>")

	if output[-1] != "</s>" and not as_lists:
		output.append("</s>")

	output = "\n".join(output)
	if as_lists:
		output = output.split("</s>")
		output = [x.strip().split("\n") for x in output if len(x.strip()) > 0]

	return output
