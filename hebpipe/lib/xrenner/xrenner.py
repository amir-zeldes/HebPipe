#!/usr/bin/python
# -*- coding: utf-8 -*-

"""
xrenner - eXternally configurable REference and Non-Named Entity Recognizer
xrenner.py
Main controller script for entity recognition and coreference resolution
Author: Amir Zeldes
"""

import argparse, sys, io, os
from modules.xrenner_xrenner import Xrenner
from glob import glob
from multiprocessing import Process, Value, Lock, current_process, Array
import ctypes
from math import ceil

__version__ = "2.1.0"
xrenner_version = "xrenner V" + __version__

sys.dont_write_bytecode = True

if sys.version_info[0] > 2:
	PY3 = True
else:
	PY3 = False
	reload(sys)
	sys.setdefaultencoding('utf8')


class Counter(object):
	"""
	Counter to synchronize progress in multiple processes. Set lock to False
	for single process, especially for running from a subprocess.
	Note that for verbose behavior, lock must be True, and subprocess invocation
	may fail.
	"""
	def __init__(self, initval=0, lock=True):
		if lock:
			self.docs = Value('i', initval)
			self.sents = Value('i', initval)
			self.toks = Value('i', initval)
			if PY3:
				self.dump_headers = Array(ctypes.c_char, b"_" * 3000)  # Value(ctypes.c_char_p, "")
			else:
				self.dump_headers = Array(ctypes.c_char,"_"*3000)# Value(ctypes.c_char_p, "")
			self.lock = Lock()
		else:
			self.docs = initval
			self.sents = initval
			self.toks = initval
			self.lock = False
			self.dump_headers = ""

	def increment(self,docs,sents,toks):
		if not self.lock:
			self.docs += docs
			self.sents += sents
			self.toks += toks
		else:
			with self.lock:
				self.docs.value += docs
				self.sents.value += sents
				self.toks.value += toks

	def set_string(self,new_val):
		if not self.lock:
			self.dump_headers = new_val
		else:
			with self.lock:
				self.dump_headers.value = new_val

	def get_string(self):
		if not self.lock:
			return self.dump_headers
		else:
			with self.lock:
				return self.dump_headers.value

	def value(self):
		if not self.lock:
			return (self.docs, self.sents, self.toks)
		else:
			with self.lock:
				return (self.docs.value,self.sents.value,self.toks.value)


def rreplace(s, old, new, occurrence):
	li = s.rsplit(old, occurrence)
	return new.join(li)

def xrenner_worker(data,options,total_docs,counter):
	tokens = 0
	sentences = 0

	model = options.model
	override = options.override
	xrenner = Xrenner(model, override, options.rulebased, options.noseq)

	if options.dump is not None:
		xrenner.lex.procid = str(current_process().ident)
		xrenner.lex.dump = io.open(rreplace(options.dump,".", xrenner.lex.procid+".",1),'w',encoding="utf8",newline="\n")
	else:
		xrenner.lex.procid = ""
		xrenner.lex.dump = None

	if options.oracle is not None:
		xrenner.lex.read_oracle(options.oracle)

	for file_ in data:

		xrenner.lex.dump_types = set([])  # Empty set of dump rows to avoid duplicates
		output = xrenner.analyze(file_, options.format)
		tokens += len(xrenner.conll_tokens)-1
		sentences += xrenner.sent_num-1

		if options.format == "none":
			pass
		elif options.format != "paula":
			if len(data) > 1:
				if options.format == "webanno":
					extension = "xmi"
				elif options.format == "webannotsv":
					extension = "tsv"
				else:
					extension = options.format
				outfile = xrenner.docname + "." + extension
				handle = io.open(outfile, 'w', encoding="utf8")
				handle.write(output)
				handle.close()
			else:
				if PY3:
					sys.stdout.buffer.write(output.encode("utf8"))
				else:
					print(output.encode("utf8"))

		counter.increment(1,xrenner.sent_num-1,len(xrenner.conll_tokens)-1)
		docs, sents, toks = counter.value()
		if options.verbose and options.oracle is not None:
			sys.stderr.write("Used oracle for " + str(xrenner.lex.oracle_counters[2]) + " entities, of which " + str(xrenner.lex.oracle_counters[0]) + " had spans in oracle and " + str(xrenner.lex.oracle_counters[1]) + " were different types than xrenner pred\n")

		if options.verbose and len(data) > 1:
			sys.stderr.write("Document " + str(docs) + "/" + str(total_docs) + ": " +
								 "Processed " + str(len(xrenner.conll_tokens)-1) + " tokens in " + str(xrenner.sent_num-1) + " sentences.\n")

	if options.dump is not None:
		xrenner.lex.dump.close()
		if PY3:
			counter.set_string(bytes("\t".join(xrenner.lex.dump_headers),encoding="utf8"))
		else:
			counter.set_string("\t".join(xrenner.lex.dump_headers))


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-o', '--output', action="store", dest="format", default="sgml", help="output format, default: sgml",
						choices=["sgml", "html", "paula", "webanno", "webannotsv", "conll", "onto", "unittest", "none"])
	parser.add_argument('-m', '--model', action="store", dest="model", default="eng", help="input model directory name, in models/")
	parser.add_argument('-x', '--override', action="store", dest="override", default=None, help="specify a section in the model's override.ini file with alternative settings")
	parser.add_argument('-r', '--rulebased', action="store_true", help="run model without machine learning classifiers")
	parser.add_argument('-v', '--verbose', action="store_true", help="output run time and summary")
	parser.add_argument('-t', '--test', action="store_true", dest="test", help="run unit tests and quit")
	parser.add_argument('-p', '--procs', type=int, choices=range(1,17), dest="procs", help="number of processes for multithreading", default=2)
	parser.add_argument('-d', '--dump', action="store", dest="dump", help="file to dump individual analyses into", default=None)
	parser.add_argument('file', action="store", help="input file name to process")
	parser.add_argument('--oracle', action='store', help="file with oracle entity predictions")
	parser.add_argument('--noseq', action='store_true', help="do not use sequence tagger for entity classification")
	parser.add_argument('--version', action='version', version=xrenner_version, help="show xrenner version number and quit")

	total_docs = 0

	# Check if -t is invoked and run unit tests instead of parsing command line
	if len(sys.argv) > 1 and sys.argv[1] in ["-t", "--test"]:
		import unittest
		import modules.xrenner_test
		suite = unittest.TestLoader().loadTestsFromModule(modules.xrenner_test)
		unittest.TextTestRunner().run(suite)
	# Not a test run, parse command line as usual
	else:
		options = parser.parse_args()
		procs = options.procs
		if options.verbose:
			import modules.timing
			sys.stderr.write("\nReading language model...\n")

		data = glob(options.file)
		if data == []:
			sys.stderr.write("\nCan't find input at " + options.file +"\nAborting\n")
			sys.exit()
		if not isinstance(data, list):
			split_data = [data]
		else:
			if len(data) < procs:  # Do not use more processes than files to process
				procs = len(data)
			chunk_size = int(ceil(len(data)/float(procs)))
			split_data = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

		if procs > 1 or options.verbose:
			lock = True
		else:
			lock = True
		counter = Counter(0,lock=lock)
		jobs = []
		dump_files = []

		if options.dump is not None:
			if "." not in options.dump:
				options.dump += ".tab"

		for sublist in split_data:
			p = Process(target=xrenner_worker, args=(sublist,options,len(data), counter))
			jobs.append(p)
			p.start()
			if options.dump is not None:
				dump_files.append(rreplace(options.dump, ".", str(p.ident)+".",1))
		for j in jobs:
			j.join()

		total_docs, total_sentences, total_tokens = counter.value()

		if options.verbose:
			sys.stderr.write("="*40 + "\n")
			sys.stderr.write("Processed " + str(total_tokens) + " tokens in " + str(total_sentences) + " sentences.\n")

		if len(dump_files) > 0:
			# Merge dump files from multiple processes
			sys.stderr.write("Collating dump data ... \n")
			with io.open(options.dump,'w',encoding="utf8",newline="\n") as wfd:
				if lock:
					if PY3:
						headers = counter.dump_headers.value.decode("utf8")
					else:
						headers = counter.dump_headers.value
				else:
					headers = counter.dump_headers
				wfd.write(headers + "\n")
			with io.open(options.dump,'a',encoding="utf8") as wfd:
				for f in dump_files:
					with io.open(f,'r', encoding="utf8") as fd:
						temp = fd.read()
						wfd.write(temp)
			for f in dump_files:
				os.remove(f)
			sys.stderr.write("Dump written to "+options.dump+"\n")

