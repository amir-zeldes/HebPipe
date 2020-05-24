"""
modules/xrenner_sequence.py

Adapter class to accommodate sequence labeler
  * Supplies a uniform predict_proba() method
  * Reads serialized models
  * Compatible with flair embeddings

Author: Amir Zeldes
"""

import sys, os

script_dir = os.path.dirname(os.path.realpath(__file__)) + os.sep

PY3 = sys.version_info[0] > 2

class StdOutFilter(object):
	def __init__(self): #, strings_to_filter, stream):
		self.stream = sys.stdout
		self.stdout = sys.stdout

	def __getattr__(self, attr_name):
		return getattr(self.stream, attr_name)

	def write(self, data):
		output = []
		lines = data.split("\n")
		for line in lines:
			if "Epoch: " in line or (" f:" in line and "Test" not in line):
				output.append(line)
		if len(output)>0:
			data = "\n".join(output) + "\n"
			self.stream.write("RNN log - " + data.strip() + "\n")
			self.stream.flush()

	def flush(self):
		pass
		#self.stream.flush()

	def start(self):
		sys.stdout = self

	def end(self):
		sys.stdout = self.stdout

p = StdOutFilter()
p.start()

# Silently import flair/torch
from flair.data import Sentence
from flair.models import SequenceTagger
import torch

p.end()


class Sequencer:
	def __init__(self, model_path=None):
		if model_path is None:
			model_path = script_dir + ".." + os.sep + "models" + os.sep + "_sequence_taggers" + os.sep + "eng_flair_nner_distilbert.pt"
		elif os.sep not in model_path:  # Assume this is a file in models/_sequence_taggers
			model_path = script_dir + ".." + os.sep + "models" + os.sep + "_sequence_taggers" + os.sep + model_path
		if not os.path.exists(model_path):
			sys.stderr.write("! Sequence tagger model file missing at " + model_path + "\n")
			sys.stderr.write("! Add the model file or use get_models.py to obtain built-in models\nAborting...\n")
			quit()
		self.tagger = SequenceTagger.load(model_path)

	def clear_embeddings(self, sentences, also_clear_word_embeddings=False):
		"""
		Clears the embeddings from all given sentences.
		:param sentences: list of sentences
		"""
		for sentence in sentences:
			sentence.clear_embeddings(also_clear_word_embeddings=also_clear_word_embeddings)

	def predict_proba(self, sentences):
		"""
		Predicts a list of class and class probability tuples for every token in a list of sentences
		:param sentences: list of space tokenized sentence strings
		:return: the list of sentences containing the labels
		"""

		# Sort sentences and keep order
		sents = [(len(s.split()),i,s) for i, s in enumerate(sentences)]
		sents.sort(key=lambda x:x[0], reverse=True)
		sentences = [s[2] for s in sents]

		preds = self.tagger.predict(sentences)

		# sort back
		sents = [tuple(list(sents[i]) + [s]) for i, s in enumerate(preds)]
		sents.sort(key=lambda x:x[1])
		sents = [s[3] for s in sents]

		output = []
		for s in sents:
			for tok in s.tokens:
				output.append((tok.tags['ner'].value.replace("S-","").replace("B-","").replace("I-","").replace("E-",""),
							   tok.tags['ner'].score))

		return output


if __name__ == "__main__":
	c = Sequencer()
	x = c.predict_proba(["Mary had a little lamb","Her fleece was white as snow .","I joined Intel in the Age of Knives"])
	print(x)
