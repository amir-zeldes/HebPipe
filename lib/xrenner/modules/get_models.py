#!/usr/bin/python
# -*- coding: utf-8 -*-
import os, requests, io, shutil, sys

script_dir = os.path.dirname(os.path.realpath(__file__))

def download_file(url, local_path):
	try:
		sys.stderr.write("o Downloading model from " + url + "...\n")
		with requests.get(url, stream=True) as r:
			with io.open(local_path, 'wb') as f:
				shutil.copyfileobj(r.raw, f)
		sys.stderr.write("o Download successful\n")
	except Exception as e:
		sys.stderr.write("\n! Could not download model from " + url + "\n")
		sys.stderr.write(str(e))


def check_models(path=None):
# Check for missing models
	if path is None:
		model = "eng_flair_nner_distilbert.pt"
		path = os.sep.join([script_dir,"..","models","_sequence_taggers",model])
	else:
		model = path.split(os.sep)[-1]

	if not os.path.exists(path):
		server = "corpling.uis.georgetown.edu"
		resource = "/".join([server, "amir", "download", model])
		download_file("https://" + resource, os.sep.join([script_dir,"..","models","_sequence_taggers",model]))
	else:
		sys.stderr.write("o Model " + model + " already found in models/_sequence_taggers/")

if __name__ == "__main__":
	check_models()
