"""
Basic classes for parsed tokens, markables and sentences.

Author: Amir Zeldes
"""

import sys, re
from math import log
from collections import OrderedDict, defaultdict

class ParsedToken:
	def __init__(self, tok_id, text, lemma, pos, morph, head, func, sentence, modifiers, child_funcs, child_strings, lex, quoted=False, head2="_", func2="_"):
		self.id = tok_id
		self.text = text.strip()
		self.text_lower = text.lower()
		self.pos = pos
		if lemma != "_" and lemma != "--":
			self.lemma = lemma.strip()
		else:
			self.lemma = lex.lemmatize(self)
		self.morph = morph
		if morph != "_" and morph != "--" and morph != "":
			self.morph = lex.process_morph(self)

		self.head = head
		self.original_head = head
		self.func = func
		self.head2 = head2
		self.func2 = func2
		self.sentence = sentence
		self.modifiers = modifiers
		self.child_funcs = child_funcs
		self.child_strings = child_strings
		self.quoted = quoted
		self.coordinate = False
		self.head_text = ""
		self.lex = lex  # Pointer to lex object
		self.lemma_freq = 0.0
		self.negated = False
		self.neg_parent = False

	def __repr__(self):
		return str(self.text) + " (" + str(self.pos) + "/" + str(self.lemma) + ") " + "<-" + str(self.func) + "- " + str(self.head_text)

class Markable:

	# Properties refering to markable head, not markable itself
	tok_props = {"negated", "neg_parent", "pos", "lemma", "morph", "func", "quoted", "modifiers", "child_funcs", "child_strings",
	 "agree", "doc_position", "sent_position", "head_text", "head_pos", "lemma_freq"}

	def __init__(self, mark_id, head, form, definiteness, start, end, text, core_text, entity, entity_certainty, subclass, infstat, agree, sentence,
				 antecedent, coref_type, group, alt_entities, alt_subclasses, alt_agree, cardinality=0, submarks=[], coordinate=False, agree_certainty=""):
		self.id = mark_id
		self.head = head
		self.form = form
		self.definiteness = definiteness
		self.start = start
		self.end = end
		self.text = text.strip()
		self.core_text = core_text.strip()  # Holds markable text before any extensions or manipulations
		self.first = self.core_text.split(" ")[0]
		self.last = self.core_text.split(" ")[-1]
		self.entity = entity
		self.subclass = subclass
		self.infstat = infstat
		self.agree = agree
		self.agree_certainty = agree_certainty
		self.sentence = sentence
		self.antecedent = antecedent
		self.coref_type = coref_type
		self.group = group
		self.non_antecdent_groups = set()
		self.entity_certainty = entity_certainty
		self.isa_partner_head = ""  # Property to hold isa match; once saturated, no other lexeme may form isa link

		# Alternate agreement, subclass and entity lists:
		self.alt_agree = alt_agree
		self.alt_entities = alt_entities
		self.alt_subclasses = alt_subclasses

		self.cardinality=cardinality
		self.submarks = submarks
		self.coordinate = coordinate

		self.length = self.text.strip().count(" ") + 1
		self.mod_count = len(self.head.modifiers)

		# Dictionaries to fill when getting dependency and similarity based frequencies
		self.entity_dep_scores = defaultdict(int)
		self.entity_sim_dep_scores = defaultdict(int)
		self.lex_dep_scores = defaultdict(int)
		self.lex_sim_dep_scores = defaultdict(int)

	def has_child_func(self, func):
		if "*" in func: # func substring, do not delimit function
			return func in self.child_func_string
		else: # Exact match, delimit with ";"
			return ";" + func + ";" in self.child_func_string

	def get_dep_freqs(self,lex):

		#DEBUG POINT
		if self.text == lex.debug["ana"]:
			pass

		use_entity_deps = True
		use_lex_deps = True
		if "ablations" in lex.debug:
			if "no_entity_dep" in lex.debug["ablations"]:
				use_entity_deps = False
		use_hasa = True
		if "ablations" in lex.debug:
			if "no_hasa" in lex.debug["ablations"]:
				use_hasa = False

		anaphor_parent = self.head.head_text
		if anaphor_parent in lex.entity_deps and use_entity_deps:
			if self.head.func in lex.entity_deps[anaphor_parent]:
				for entity in lex.entity_deps[anaphor_parent][self.head.func]:
					self.entity_dep_scores[entity] = lex.entity_deps[anaphor_parent][self.head.func][entity]
		found = False
		if anaphor_parent in lex.similar and use_entity_deps:
			for sim in lex.similar[anaphor_parent]:
				if sim in lex.entity_deps and not found:
					if self.head.func in lex.entity_deps[sim]:
						found = True
						for entity in lex.entity_deps[sim][self.head.func]:
							self.entity_sim_dep_scores[entity] = lex.entity_deps[sim][self.head.func][entity]
		if anaphor_parent in lex.lex_deps and use_lex_deps:
			if self.head.func in lex.lex_deps[anaphor_parent]:
				for lexeme in lex.lex_deps[anaphor_parent][self.head.func]:
					self.lex_dep_scores[lexeme] = lex.lex_deps[anaphor_parent][self.head.func][lexeme]
		if anaphor_parent in lex.similar and use_entity_deps:
			for sim in lex.similar[anaphor_parent]:
				if sim in lex.lex_deps:
					if self.head.func in lex.lex_deps[sim]:
						for lexeme in lex.lex_deps[sim][self.head.func]:
							self.entity_sim_dep_scores[lexeme] = lex.lex_deps[sim][self.head.func][lexeme]

	def __repr__(self):
		agree = "no-agr" if self.agree == "" else self.agree
		defin = "no-def" if self.definiteness == "" else self.definiteness
		card = "no-card" if self.cardinality == 0 else self.cardinality
		func = "no-func" if self.head.func == "" else self.head.func
		return str(self.entity) + "/" + str(self.subclass) + ': "' + self.text + '" (' + agree + "/" + defin + "/" + func + "/" + str(card) + ")"


	def extract_features(self, lex, antecedent=None, candidate_list=[], dump_position=False):
		"""
		Function to generate feature representation of markables or markable-antecedent pairs for classifiers

		:param lex: the LexData object with gazetteer information and model settings
		:param antecedent: The antecedent Markable potentially coreferring to self
		:param candidate_list: The list of candidate markables under consideration, used to extract cohort size
		:param dump_position: Whether document name + token positions are dumped for each markable to compare to gold
		:return: dictionary of markable properties
		"""

		out_dict = OrderedDict()
		if dump_position:
			out_dict["position"] = str(self.start)+"-"+str(self.end)+";"+str(antecedent.start)+"-"+str(antecedent.end)
		out_dict["docname"] = lex.docname

		# TODO: Make genre representation configurable
		# By convention, genre is taken from first 4 chars of file, but for corpora like GUM use part between underscores
		if lex.docname.startswith("GUM_") or lex.docname.lower().startswith("autogum_") or lex.docname.lower().startswith("amalgum_"):
			out_dict["genre"] = lex.docname.split("_")[1]
		elif len(lex.docname) > 4:
			out_dict["genre"] = lex.docname[:4]
		else:
			out_dict["genre"] = "_"  # Too short to detect genre

		log_props = set([])  # Properties to log-transform before dump, if desired
		bool_props = {"coordinate","quoted","negated","neg_parent"}  # Boolean props to dump as 1 or 0
		thresh_props = {"lemma","head_text"}
		f_threshold = 0  # Minimum frequency for lexical categories in freqs.tab (0 to ignore, else items with lower freq are replace by POS)

		# TODO: May need to lower() head_text, since cap/uncapped version may be in lex/entity dep score lexicon
		anaphor_parent = self.head.head_text
		ana_props = ["lemma","func","head_text","form","pos","agree","start","end","lemma_freq",
						  "cardinality", "definiteness","entity","subclass", "infstat", "coordinate",
						  "length", "mod_count", "doc_position","sent_position","quoted","negated","neg_parent","s_type"]
		ante_props = ["lemma","func","head_text","form","pos","agree","start","end","lemma_freq",
						  "cardinality", "definiteness","entity","subclass", "infstat", "coordinate",
						  "length", "mod_count", "doc_position","sent_position","quoted","negated","neg_parent","s_type"]
		for prop in ana_props:
			val = getattr(self,prop)
			if prop in log_props:
				val = log(val+1)
			elif prop in bool_props:
				val = int(val)
			elif prop in ["lemma"]:
				if lex.freqs[val] < f_threshold:
					val = self.pos
			elif prop in ["head_text"]:
				if lex.freqs[val] < f_threshold:
					val = self.head_pos
			out_dict["n_" + prop] = val if val != "" else "_"

		if antecedent is not None:
			for prop in ante_props:
				val = getattr(antecedent,prop)
				if prop in log_props:
					val = log(val + 1)
				elif prop in bool_props:
					val = int(val)
				elif prop in thresh_props:
					if lex.freqs[val] < f_threshold:
						if prop == "lemma":
							val = antecedent.pos
						elif prop == "head_text":
							val = antecedent.head_pos
				out_dict["t_" + prop] = val if val != "" else "_"

			anaphor = self

			out_dict["d_sent"] = anaphor.sent_num - antecedent.sent_num
			out_dict["d_tok"] = anaphor.start - antecedent.end
			out_dict["d_agr"] = int(anaphor.agree == antecedent.agree)
			out_dict["d_intervene"] = abs(int(re.sub('.*_', '', anaphor.id)) - int(re.sub('.*_', '', antecedent.id)))  # Number of markables between ana and ante
			out_dict["d_cohort"] = len(candidate_list)
			out_dict["d_modcount"] = anaphor.mod_count - antecedent.mod_count
			out_dict["d_samemods"] = len(list(set([m.lemma for m in anaphor.head.modifiers]) & set([m.lemma for m in antecedent.head.modifiers])))

			hasa = 0
			use_hasa = True
			if "ablations" in lex.debug:
				if "no_hasa" in lex.debug["ablations"]:
					use_hasa = False
			if use_hasa:
				if antecedent.head.text in lex.hasa and lex.filters["possessive_func"].search(self.func) is not None:  # Text based hasa
					if anaphor_parent in lex.hasa[antecedent.head.text]:
						hasa = lex.hasa[antecedent.head.text][anaphor_parent]
				elif antecedent.head.lemma in lex.hasa and lex.filters["possessive_func"].search(self.func) is not None:  # Lemma based hasa
					if anaphor_parent in lex.hasa[antecedent.head.lemma]:
						hasa = lex.hasa[antecedent.head.lemma][anaphor_parent]

			out_dict["d_hasa"] = hasa
			out_dict["d_entidep"] = self.entity_dep_scores[antecedent.entity]
			out_dict["d_entisimdep"] = self.entity_sim_dep_scores[antecedent.entity]
			out_dict["d_lexdep"] = self.lex_dep_scores[antecedent.head.text]
			out_dict["d_lexsimdep"] = self.lex_sim_dep_scores[antecedent.head.text]
			out_dict["d_sametext"] = int(anaphor.text == antecedent.text)
			out_dict["d_samelemma"] = int(anaphor.lemma == antecedent.lemma)

			out_dict["d_doclen"] = int(anaphor.head.lex.token_count)


			# Check if one markable head is the dependency parent of the other
			if antecedent.head.head == anaphor.head.id:
				out_dict["d_parent"] = 1
			elif anaphor.head.head == antecedent.head.id:
				out_dict["d_parent"] = -1
			else:
				out_dict["d_parent"] = 0

			if self.speaker == "" and antecedent.speaker == "":
				out_dict["d_speaker"] = 0
			elif self.speaker == antecedent.speaker:
				out_dict["d_speaker"] = 1
			else:
				out_dict["d_speaker"] = -1

		return out_dict

	def __getattr__(self, item):
		# Convenience methods to access head token and containing sentence
		if item in self.tok_props:
			return getattr(self.head,item)
		elif item == "text_lower":
			if self.coordinate:  # If this is a coordinate markable return lower case core_text
				return self.core_text.lower()
			else:  # Otherwise return lower text of head token
				return getattr(self.head, item)
		elif item in ["mood", "speaker", "sent_num", "s_type"]:
			return getattr(self.sentence,item)
		elif item == "child_func_string":
			# Check for cached child_func_string
			if "child_func_string" not in self.__dict__:  # Convenience property to store semi-colon separated child funcs of head token
				# Assemble if not yet cached
				if len(self.head.child_funcs) > 1:
					self.child_func_string = ";" + ";".join(self.head.child_funcs) + ";"
				else:
					self.child_func_string = "_"
			return self.child_func_string
		else:
			raise AttributeError("No attribute: " + str(item))


class Sentence:
	def __init__(self, sent_num, start_offset, mood="", speaker=""):
		self.sent_num = sent_num
		self.start_offset = start_offset
		self.mood = mood
		self.speaker = speaker
		self.token_count = 0
		self.s_type = "_"  # Initial type, will be overwritten if s_type annotation is found in input

	def __repr__(self):
		mood = "(no mood info)" if self.mood == "" else self.mood
		speaker = "(no speaker info)" if self.speaker == "" else self.speaker
		return "S" + str(self.sent_num) + " from T" + str(self.start_offset + 1) + ", mood: " + mood  + ", speaker: " + speaker + ", type: " + self.s_type


def get_descendants(parent, children_dict, seen_tokens, sent_num, conll_tokens):
	my_descendants = []
	my_descendants += children_dict[parent]
	for child in children_dict[parent]:
		if child in seen_tokens:
			if sys.version_info[0] < 3:
				sys.stderr.write("\nCycle detected in syntax tree in " + conll_tokens[int(parent)].lex.docname + " in sentence " + str(sent_num) + " (child of token: '" + conll_tokens[int(parent)].text.encode("utf8") + "')\n")
			else:
				sys.stderr.write("\nCycle detected in syntax tree in " + conll_tokens[int(parent)].lex.docname + " in sentence " + str(sent_num) + " (child of token: '" + conll_tokens[int(parent)].text + "')\n")
			sys.exit("Exiting due to invalid input\n")
		else:
			seen_tokens += [child]
	for child in children_dict[parent]:
		if child in children_dict:
			my_descendants += get_descendants(child, children_dict, seen_tokens, sent_num, conll_tokens)
	return my_descendants
