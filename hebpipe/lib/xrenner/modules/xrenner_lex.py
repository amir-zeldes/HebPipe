# -*- coding: utf-8 -*-

import gc
import os
from os import listdir
from os.path import isfile, join
import re
import sys, io
from collections import defaultdict
from .xrenner_rule import CorefRule

if sys.version_info[0] < 3:
	# Python 2
	PY2 = True
	from ConfigParser import ConfigParser, NoSectionError
	def unicode_split_reader(f):
		return [line.replace("\n","").replace('\\"','"').split('\t') for line in f.read().strip().split("\n")]
		#return [line.replace("\n","").replace('\\"','"').split('\t') for line in f.readlines()]
else:
	# Python 3
	PY2 = False
	from configparser import NoSectionError, RawConfigParser as ConfigParser
	import csv

"""
LexData class - container object for lexical information, gazetteers etc.

Author: Amir Zeldes
"""

class LexData:
	"""
	Class to hold lexical information from gazetteers and training data.
	Use model argument to define subdirectory under models/ for reading different sets of
	configuration files.
	"""

	def __init__(self, model, xrenner, override=None, rule_based=False, no_seq=False):
		"""
		:param model: model - string name of the model to read from models/
		:param override: override - optional name of a section to use in models/override.ini
		:param rule_based: do not use machine learning classifiers for coreference resolution
		:param no_seq: do not use machine learning sequence taggers for entity resolution
		"""
		gc.disable()
		self.model = model
		self.docname = "untitled"
		self.atoms = {}
		self.mod_atoms = {}
		self.dump = None  # Placeholder for data dump destination
		self.dump_headers = []  # Placeholder for data dump feature names
		self.classifiers = {}  # Holds loaded classifiers from pickled files in model
		self.xrenner = xrenner
		self.entity_oracle = None  # Holds external entity predictions to use instead of system predictions
		self.oracle_counters = [0,0,0]
		self.lemma_freqs = defaultdict(float)  # Holds proportional frequency of each lemma in document
		self.token_count = 0  # Holds copy of token count for document from Xrenner object

		# Lookup model path

		if os.sep in self.model:  # Check if model provided is an absolute or relative path
			model_path = os.path.abspath(self.model)
		# Check the unlikely case that the model is directly in xrenner.py's directory
		elif os.path.exists(os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." + os.sep + model):
			model_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." + os.sep + model
		else:  # This is a pathless model name specification, which is expected to live in ./models/
			model_path = os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." + os.sep + "models" + os.sep + model

		if not os.path.exists(model_path):  # Check if this is a default model name that might need the .xrm extension
			if os.path.exists(model_path + ".xrm"):
				model_path += ".xrm"
			else:
				raise IOError("Model not found at: " + model_path)

		# Read model files and store in self.model_files dictionary

		self.model_files = {}
		if os.path.isdir(model_path):  # Check if this is a directory or a compressed model file
			model_path += os.sep
			model_files_list = [f for f in listdir(model_path) if isfile(join(model_path, f))]
			for filename in model_files_list:
				if filename.endswith(".pkl"):  # Classifier
					if not (filename.endswith("3.pkl") and PY2) and not (filename.endswith("2.pkl") and not PY2):  # Skip unpickling unneeded PY2/3 classifiers
						self.model_files[filename] = open(model_path + filename, 'rb')
				else:
					self.model_files[filename] = io.open(model_path + filename, 'r', encoding="utf8")
		else:
			from zipfile import ZipFile
			try:
				zip = ZipFile(model_path)
				model_files_list = [f for f in zip.namelist() if not os.path.isdir(f)]
				for filename in model_files_list:
					if sys.version_info[0] < 3 or filename.endswith(".pkl"):  # Python 2 or classifier
						if not (filename.endswith("3.pkl") and PY2) and not (filename.endswith("2.pkl") and not PY2):  # Skip unpickling unneeded PY2/3 classifiers
							self.model_files[filename] = zip.open(filename)
					else:
						self.model_files[filename] = io.TextIOWrapper(zip.open(filename), encoding="utf8")
			except:
				raise IOError("Could not open model file " + filename)


		self.entity_sums = defaultdict(int)

		# Get configuration
		self.filters = self.get_filters(override)
		if rule_based:
			self.filters["use_classifiers"] = False

		# Mandatory files must be included in model
		self.speaker_rules, self.non_speaker_rules = self.parse_coref_rules(self.model_files['coref_rules.tab'].read().replace("\r","").split("\n"))
		self.coref_rules = self.non_speaker_rules
		self.entities = self.read_delim(self.model_files['entities.tab'], 'quadruple') if 'entities.tab' in self.model_files else {}
		self.entity_heads = self.read_delim(self.model_files['entity_heads.tab'], 'quadruple', 'atoms', True) if 'entity_heads.tab' in self.model_files else {}
		self.pronouns = self.read_delim(self.model_files['pronouns.tab'], 'double') if "pronouns.tab" in self.model_files else {}

		# Optional files improve model accuracy
		self.names = self.read_delim(self.model_files['names.tab']) if "names.tab" in self.model_files else {}
		self.stop_list = self.read_delim(self.model_files['stop_list.tab'], 'low') if "stop_list.tab" in self.model_files else set([])
		self.open_close_punct = self.read_delim(self.model_files['open_close_punct.tab']) if "open_close_punct.tab" in self.model_files else {}
		self.open_close_punct_rev = dict((v, k) for k, v in self.open_close_punct.items())
		self.entity_mods = self.read_delim(self.model_files['entity_mods.tab'], 'triple', 'mod_atoms') if "entity_mods.tab" in self.model_files else {}
		self.entity_deps = self.read_delim(self.model_files['entity_deps.tab'], 'quadruple_numeric') if "entity_deps.tab" in self.model_files else {}
		self.lex_deps = self.read_delim(self.model_files['lex_deps.tab'], 'quadruple_numeric') if "lex_deps.tab" in self.model_files else {}
		self.hasa = self.read_delim(self.model_files['hasa.tab'], 'triple_numeric') if "hasa.tab" in self.model_files else defaultdict(lambda: defaultdict(int))
		self.coref = self.read_delim(self.model_files['coref.tab']) if "coref.tab" in self.model_files else {}
		self.numbers = self.read_delim(self.model_files['numbers.tab'], 'double') if "numbers.tab" in self.model_files else {}
		self.affix_tokens = self.read_delim(self.model_files['affix_tokens.tab']) if "affix_tokens.tab" in self.model_files else {}
		self.antonyms = self.read_antonyms() if "antonyms.tab" in self.model_files else {}
		self.isa = self.read_isa() if "isa.tab" in self.model_files else {}
		self.similar = self.read_delim(self.model_files['similar.tab'], 'double_with_sep') if "similar.tab" in self.model_files else {}
		self.nominalizations = self.read_delim(self.model_files['nominalizations.tab'], 'triple_numeric') if "nominalizations.tab" in self.model_files else {}
		self.debug = self.read_delim(self.model_files['debug.tab']) if "debug.tab" in self.model_files else {"ana":"","ante":"","ablations":""}
		self.freqs = self.read_delim(self.model_files['freqs.tab'],'double_numeric') if "freqs.tab" in self.model_files else defaultdict(int)
		additional_atoms = self.read_delim(self.model_files['atoms.tab'], 'double') if "atoms.tab" in self.model_files else {}

		# Compile atom and first + last name data
		self.atoms = self.get_atoms()
		self.atoms.update(additional_atoms)
		self.first_names, self.last_names = self.get_first_last_names(self.names)

		if self.filters["no_new_modifiers"] and self.filters["use_new_modifier_exceptions"]:
			self.exceptional_new_modifiers = self.read_delim(self.model_files['new_modifiers.tab'], 'double') if "new_modifiers.tab" in self.model_files else {}
			self.exceptional_new_modifiers.update(dict((name, 1) for name in self.first_names))
		else:
			self.exceptional_new_modifiers = {}


		self.pos_agree_mappings = self.get_pos_agree_mappings()
		self.last = {}

		self.morph = self.get_morph()
		self.func_substitutes_forward, self.func_substitutes_backward = self.get_func_substitutes()

		# Compile lemmatizer and morph rules
		self.lemma_rules = self.compile_lemmatization()
		self.morph_rules = self.compile_morph_rules()

		# Parse nested entity removal types
		self.rm_nested_entities = self.parse_rm_nested_entities()

		# Caching lists for already established non-matching pairs
		self.incompatible_mod_pairs = set([])
		self.incompatible_isa_pairs = set([])

		# Load sequence classifier if specified
		self.sequencer = None
		if "sequencer" in self.filters and not no_seq:
			if len(self.filters["sequencer"]) > 0:
				from .xrenner_sequence import Sequencer
				self.sequencer = Sequencer(model_path=self.filters["sequencer"])

		gc.enable()

	def read_delim(self, filename, mode="normal", atom_list_name="atoms", add_to_sums=False, sep=","):
		"""
		Generic file reader for lexical data in model directory

		:param filename: string - name of the file
		:param mode: double, triple, quadruple, quadruple_numeric, triple_numeric or low reading mode
		:param atom_list_name: list of atoms to use for triple reader mode
		:param add_to_sums: whether to sum numbers from multiple instances of the same key
		:param sep: separator for double_with_sep mode
		:return: compiled lexical data, usually a structured dictionary or set depending on number of columns
		"""
		if atom_list_name == "atoms":
			atom_list = self.atoms
		elif atom_list_name == "mod_atoms":
			atom_list = self.mod_atoms
		with filename as csvfile:
			if PY2:
				reader = unicode_split_reader(csvfile)
			else:
				reader = csv.reader(csvfile, delimiter='\t', escapechar="\\", quoting=csv.QUOTE_NONE)
			if mode == "low":
				return set([rows[0].lower() for rows in reader if not rows[0].startswith('#') and not len(rows[0]) == 0])
			elif mode == "double":
				out_dict = {}
				for rows in reader:
					if not rows[0].startswith('#') and not len(rows[0]) == 0:
						if rows[0] in out_dict:
							out_dict[rows[0]].append(rows[1])
						else:
							out_dict[rows[0]] = [rows[1]]
				return out_dict
			elif mode == "triple":
				out_dict = {}
				for rows in reader:
					if not rows[0].startswith('#'):
						if rows[2].endswith('@'):
							rows[2] = rows[2][0:-1]
							atom_list[rows[0]] = rows[1]
						if add_to_sums:
							self.entity_sums[rows[1]] += 1
						if rows[0] in out_dict:
							out_dict[rows[0]].append(rows[1] + "\t" + rows[2])
						else:
							out_dict[rows[0]] = [rows[1] + "\t" + rows[2]]
				return out_dict
			elif mode == "quadruple":
				out_dict = {}
				for rows in reader:
					if not rows[0].startswith('#'):
						if rows[2].endswith('@'):
							rows[2] = rows[2][0:-1]
							atom_list[rows[0]] = rows[1]
						if add_to_sums:
							self.entity_sums[rows[1]] += 1
						if len(rows) < 4:
							rows.append("0")
						if rows[0] in out_dict:
							out_dict[rows[0]].append(rows[1] + "\t" + rows[2] + "\t" + rows[3])
						else:
							out_dict[rows[0]] = [rows[1] + "\t" + rows[2] + "\t" + rows[3]]
				return out_dict
			elif mode == "double_numeric":
				out_dict = defaultdict(int)
				for row in reader:
					if not row[0].startswith("#"):
						out_dict[row[0]] = int(row[1])
				return out_dict
			elif mode == "triple_numeric":
				out_dict = defaultdict(lambda: defaultdict(int))
				for row in reader:
					if not row[0].startswith("#"):
						out_dict[row[0]][row[1]] = int(row[2])
				return out_dict
			elif mode == "quadruple_numeric":
				out_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(str)))
				for row in reader:
					if not row[0].startswith("#"):
						out_dict[row[0]][row[1]][row[2]] = int(row[3])
				return out_dict
			elif mode == "double_with_sep":
				out_dict = {}
				for row in reader:
					if not row[0].startswith("#"):
						out_dict[row[0]] = row[1].split(sep)
				return out_dict
			else:
				return dict((rows[0], rows[1]) for rows in reader if not rows[0].startswith('#'))

	def get_atoms(self):
		"""
		Function to compile atom list for atomic markable recognition. Currently treats listed persons, places,
		organizations and inanimate objects from lexical data as atomic by default.

		:return: dictionary of atoms.
		"""
		atoms = self.atoms
		places = dict((key, value[0]) for key, value in self.entities.items() if value[0].startswith(self.filters["place_def_entity"]+"\t"))
		atoms.update(places)
		atoms.update(self.names)
		persons = dict((key, value[0]) for key, value in self.entities.items() if value[0].startswith(self.filters["person_def_entity"]+"\t"))
		atoms.update(persons)
		organizations = dict((key, value[0]) for key, value in self.entities.items() if value[0].startswith(self.filters["organization_def_entity"]+"\t"))
		atoms.update(organizations)
		objects = dict((key, value[0]) for key, value in self.entities.items() if value[0].startswith(self.filters["object_def_entity"]+"\t"))
		atoms.update(objects)
		return atoms

	@staticmethod
	def get_first_last_names(names):
		"""
		Collects separate first and last name data from the collection in names.tab

		:param names: The complete names dictionary from names.tab, mapping full name to agreement
		:return: [firsts, lasts] - list containing dictionary of first names to agreement and set of last names
		"""
		firsts = {}
		lasts = set([])
		for name in names:
			if " " in name:
				parts = name.split(" ")
				firsts[parts[0]] = names[name]  # Get heuristic gender for this first name
				lasts.add(parts[len(parts)-1])  # Last name is a set, no gender info
		return [firsts,lasts]

	def read_antonyms(self):
		"""
		Function to created dictionary from each word to all its antonyms in antonyms.tab

		:return: dictionary from words to antonym sets
		"""
		set_list = self.read_delim(self.model_files['antonyms.tab'], 'low')
		output = defaultdict(set)
		for antoset in set_list:
			members = antoset.lower().split(",")
			for member in members:
				output[member].update(members)
				output[member].remove(member)
		return output

	def read_isa(self):
		"""
		Reads isa.tab into a dictionary from words to lists of isa-matches

		:return: dictionary from words to lists of corresponding isa-matches
		"""
		isa_list = self.read_delim(self.model_files['isa.tab'])
		output = {}
		for isa in isa_list:
			output[isa] = []
			members = isa_list[isa].split(",")
			for member in members:
				output[isa].append(member.lower())
		return output

	def get_filters(self, override=None):
		"""
		Reads model settings from config.ini and possibly overrides from override.ini

		:param override: optional section name in override.ini
		:return: filters - dictionary of settings from config.ini with possible overrides
		"""

		# e.g. override = 'GUM'
		config = ConfigParser()

		config.readfp(self.model_files["config.ini"])
		filters = defaultdict(str)

		# Set up default values for settings from newer versions for backwards compatibility
		filters["neg_func"] = re.compile("$^")
		filters["non_extend_pos"] = re.compile("$^")
		filters["core_infixes"] = re.compile("$^")
		filters["score_thresh"] = 0.5

		options = config.options("main")

		if override:
			config_ovrd = ConfigParser()
			config_ovrd.readfp(self.model_files['override.ini'])
			try:
				options_ovrd = config_ovrd.options(override)
			except NoSectionError:
				sys.stderr.write("\nNo section " + override + " in override.ini in model " + self.model + "\n")
				sys.exit()

		for option in options:
			if override and option in options_ovrd:
				try:
					option_string = config_ovrd.get(override, option)
					if option_string == -1:
						pass
					else:
						if option_string.startswith("/") and option_string.endswith("/"):
							option_string = option_string[1:-1]
							filters[option] = re.compile(option_string)
						elif option_string == "True" or option_string == "False":
							filters[option] = config_ovrd.getboolean(override, option)
						elif option_string.isdigit():
							filters[option] = config_ovrd.getint(override, option)
						elif option_string.count(".") == 1 and option_string.replace(".","").isdigit():
							filters[option] = config_ovrd.getfloat(override, option)
						else:
							filters[option] = option_string
				except:
					print("exception on %s!" % option)
					filters[option] = None
				continue
			try:
				option_string = config.get("main", option)
				if option_string == -1:
					pass
				else:
					if option_string.startswith("/") and option_string.endswith("/"):
						option_string = option_string[1:-1]
						filters[option] = re.compile(option_string)
					elif option_string == "True" or option_string == "False":
						filters[option] = config.getboolean("main", option)
					elif option_string.isdigit():
						filters[option] = config.getint("main", option)
					elif option_string.count(".") == 1 and option_string.replace(".","").isdigit():
						filters[option] = config.getfloat("main", option)
					else:
						filters[option] = option_string
			except AttributeError:
				print("exception on %s!" % option)
				filters[option] = None

		if ">" in filters["agree_entity_mapping"]:
			mappings = filters["agree_entity_mapping"].split(";")
			ent_map = {}
			for mapping in mappings:
				key, val = mapping.split(">")
				ent_map[key] = val
			filters["agree_entity_mapping"] = ent_map
		else:
			filters["agree_entity_mapping"] = {}

		return filters

	def compile_lemmatization(self):
		compiled_rules = []
		for rule in self.filters["lemma_rules"].split(";"):
			rule_part = rule.split("/")
			# Add a compiled lemmatization rule: regex matcher for pos, regex matcher for text, and replacement string
			compiled_rules.append((re.compile(rule_part[0]),re.compile(rule_part[1]),rule_part[2]))
		return compiled_rules

	def compile_morph_rules(self):
		compiled_rules = []
		for rule in self.filters["morph_rules"].split(";"):
			rule_part = rule.split("/")
			# Add a compiled lemmatization rule: regex matcher for pos, regex matcher for text, and replacement string
			if not rule[0] == rule[1]:
				compiled_rules.append((re.compile(rule_part[0]),rule_part[1]))
		return compiled_rules

	def lemmatize(self, token):
		"""
		Simple lemmatization function using rules from lemma_rules in config.ini

		:param token: ParsedToken object to be lemmatized
		:return: string - the lemma
		"""

		lemma = token.text
		for rule in self.lemma_rules:
			if rule[0].search(token.pos) is not None:
				lemma = rule[1].sub(rule[2], lemma)
		if self.filters["auto_lower_lemma"] == "all":
			return lemma.lower()
		elif self.filters["auto_lower_lemma"] == "except_all_caps":
			if lemma.upper() == lemma:
				return lemma
			else:
				return lemma.lower()
		else:
			return lemma

	def get_func_substitutes(self):
		"""
		Function for semi-hard-wired function substitutions based on function label and dependency direction.
		Uses func_substitute_forward and func_substitute_backward settings in config.ini

		:return: list of compiled substitutions_forward, substitutions_backward
		"""

		substitutions_forward = {}
		substitutions_backward = {}
		subst_rules = self.filters["func_substitute_forward"]
		for rule in subst_rules.split(";"):
			rule_part = rule.split("/")
			substitutions_forward[rule_part[0]] = [rule_part[1],rule_part[2]]
		subst_rules = self.filters["func_substitute_backward"]
		for rule in subst_rules.split(";"):
			rule_part = rule.split("/")
			substitutions_backward[rule_part[0]] = [rule_part[1],rule_part[2]]
		return [substitutions_forward,substitutions_backward]

	def process_morph(self, token):
		"""
		Simple mechanism for substituting values in morph feature of input tokens. For more elaborate sub-graph
		dependent manipultations, use depedit module

		:param token: ParsedToken object to edit morph feature
		:return: string - the edited morph feature
		"""

		morph = token.morph
		for rule in self.morph_rules:
			morph = rule[0].sub(rule[1], morph)
		return morph

	def get_pos_agree_mappings(self):
		"""
		Gets dictionary mapping POS categories to default agreement classes, e.g. NNS > plural

		:return: mapping dictionary
		"""

		mappings = {}
		rules = self.filters["pos_agree_mapping"]
		for rule in rules.split(";"):
			if ">" in rule:
				mappings[rule.split(">")[0]] = rule.split(">")[1]

		return mappings

	def parse_coref_rules(self,rule_list):
		"""
		Reader function to pass coref_rules.tab into CorefRule objects in two lists: one for general rules and
		one also including rules to use when speaker info is available.

		:param rule_list: textual list of rules
		:return: two separate lists of compiled CorefRule objects with and without speaker specifications
		"""

		rule_num = 0
		speaker_rules=[]
		non_speaker_rules=[]
		rule_list = [rule for rule in rule_list if len(rule) > 0 and not rule.startswith("#")]
		for rule in rule_list:
			rule_num += 1
			speaker_rules.append(CorefRule(rule, rule_num))
			if "speaker" not in rule:
				non_speaker_rules.append(CorefRule(rule, rule_num))

		# Load classifiers if available
		for rule in speaker_rules + non_speaker_rules:
			if rule.thresh is None:
				rule.thresh = self.filters["score_thresh"]
			if rule.clf_name != "_default_" and self.filters["use_classifiers"]:
				if self.filters["classifier_suffix"] != "":  # Add suffixes (e.g. for different versions in model override)
					rule.clf_name = rule.clf_name.replace(".pkl", self.filters["classifier_suffix"] + ".pkl")
				if rule.clf_name not in self.classifiers:
					rule_file_name = rule.clf_name
					if rule.clf_name not in self.model_files:  # File name missing, possible Python 2/3 variants available
						if sys.version_info[0] < 3:
							if rule.clf_name.replace(".pkl","2.pkl") in self.model_files:
								rule_file_name = rule.clf_name.replace(".pkl","2.pkl")
							else:
								if rule.clf_name.replace(".pkl", "3.pkl") in self.model_files:
									print("This model supports classifiers for Python 3 only.\n  * switch to Python 3 and try running again\n  * alternatively switch off classifiers with the option -r (expect lower accuracy)")
									sys.exit()
						elif sys.version_info[0] > 2:
							if rule.clf_name.replace(".pkl","3.pkl") in self.model_files:
								rule_file_name = rule.clf_name.replace(".pkl","3.pkl")
							else:
								if rule.clf_name.replace(".pkl", "2.pkl") in self.model_files:
									print("This model supports classifiers for Python 2 only.\n  * switch to Python 2 and try running again\n  * alternatively switch off classifiers with the option -r (expect lower accuracy)")
									sys.exit()
					try:
						from joblib import load
					except Exception as e:
						print("Unable to import joblib:\n  * classifiers in this model require installing joblib (pip install joblib)\n  * alternatively switch off classifiers with the option -r (expect lower accuracy)")
						sys.exit()
					from .xrenner_classify import Classifier

					try:
						clf = load(self.model_files[rule_file_name])
					except KeyError:
						print("\nClassifier '" + rule.clf_name + "' was not found in the model - check coref_rules.tab")
						sys.exit()
					clf = Classifier(clf[0], clf[1], clf[2])
					self.classifiers[rule.clf_name] = clf

		return speaker_rules, non_speaker_rules

	def parse_rm_nested_entities(self):
		rm_string = self.filters["remove_nested_entities"]
		types = rm_string.split(";")
		rm_nested_entities = []
		for ent_type in types:
			if ent_type.count(",") == 2:
				nested, func, container = ent_type.split(",")
				rm_nested_entities.append((nested, func, container))
		return rm_nested_entities

	def get_morph(self):
		"""
		Compiles morphlogical affix dictionary based on members of entity_heads.tab

		:return: dictionary from affixes to dictionaries mapping classes to type frequencies
		"""
		morph = {}
		for head in self.entity_heads:
			for i in range(1, self.filters["max_suffix_length"]):
				if len(head) > i:
					substring = head[len(head)-i:]
					entity_list = self.entity_heads[head]
					if substring in morph:
						for entity in entity_list:
							entity_class = entity.split("\t")[0]
							if entity_class in morph[substring]:
								morph[substring][entity_class] += 1
							else:
								morph[substring][entity_class] = 1
					else:
						for entity in entity_list:
							entity_class = entity.split("\t")[0]
							morph[substring] = {entity_class:1}
		return morph

	def read_oracle(self, oracle_file, as_text=True):

		self.entity_oracle = defaultdict(lambda : defaultdict(str))
		if not as_text:
			oracle_file = io.open(oracle_file,encoding="utf8").read()
		sents = oracle_file.strip().split("\n\n")

		for sent in sents:
			parts = sent.strip().split("\n")
			if len(parts) == 3:
				text = parts[0]
				preds = parts[-1]
				for pred in preds.split("|"):
					toks, entity = pred.split()
					start, end = toks.split(",")
					end = int(end) - 1
					start = int(start)
					self.entity_oracle[text][(start,end)] = entity


