#!/usr/bin/python
# -*- coding: utf-8 -*-

import re
from collections import defaultdict, OrderedDict
from .xrenner_classes import Markable
from six import iteritems, iterkeys

"""
Marker module for markable entity recognition. Establishes compatibility between entity features
and determines markable extension in tokens

Author: Amir Zeldes
"""


def is_atomic(mark, atoms, lex):
	"""
	Checks if nested markables are allowed within this markable
	
	:param mark: the :class:`.Markable` to be checked for atomicity
	:param atoms: list of atomic markable text strings
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: bool
	"""

	marktext = mark.text.strip()

	# Do not accept a markable [New] within atomic [New Zealand]
	if marktext in atoms:
		return True
	elif marktext.lower() in atoms:
		return True
	# Remove possible prefix tokens to reject [The [United] Kingdom] if [United Kingdom] in atoms
	elif remove_prefix_tokens(marktext, lex).strip() in atoms:
		return True
	# Remove possible suffix tokens to reject [[New] Zealand 's] is [New Zealand] in atoms
	elif remove_suffix_tokens(marktext, lex).strip() in atoms:
		return True
	elif remove_infix_tokens(marktext, lex).strip() in atoms:
		return True
	# Combination of prefix and suffix to reject [The [United] Kingdom 's]
	elif mark.core_text in atoms:
		return True
	elif replace_head_with_lemma(mark) in atoms:
		return True
	# Dynamic generation of proper name pattern
	elif 0 < marktext.strip().count(" ") < 3 and marktext.strip().split(" ")[0] in lex.first_names and marktext.strip().split(" ")[-1] in lex.last_names:
		return True
	else:
		non_essential_modifiers = list(mod.text for mod in mark.head.modifiers if lex.filters["non_essential_mod_func"].match(mod.func))
		if len(non_essential_modifiers) > 0:
			mark_unmod_text = mark.core_text
			for mod in non_essential_modifiers:
				mark_unmod_text = mark_unmod_text.replace(mod+" ","")
			if mark_unmod_text in lex.atoms:
				return True
		# Not an atom, nested markables allowed
		return False


def remove_suffix_tokens(marktext, lex):
	"""
	Remove trailing tokens such as genitive 's and other tokens configured as potentially redundant to citation form

	:param marktext: the markable text string to remove tokens from
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: potentially truncated text
	"""

	if lex.filters["core_suffixes"].search(marktext):
		return lex.filters["core_suffixes"].sub(" ", marktext)
	else:
		tokens = marktext.split(" ")
		suffix_candidate = ""

		for token in reversed(tokens):
			suffix_candidate = token + " " + suffix_candidate
			if suffix_candidate.strip() in lex.affix_tokens:
				if lex.affix_tokens[suffix_candidate.strip()] == "prefix":
					return re.sub(suffix_candidate + r'$', "", marktext)
	return marktext


def remove_prefix_tokens(marktext, lex):
	"""
	Remove leading tokens such as articles and other tokens configured as potentially redundant to citation form

	:param marktext: the markable text string to remove tokens from
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: potentially truncated text
	"""

	if lex.filters["core_prefixes"].match(marktext): # NB use initial match here
		return lex.filters["core_prefixes"].sub(" ", marktext)
	else:
		tokens = marktext.split(" ")
		prefix_candidate = ""
		for token in tokens:
			prefix_candidate += token + " "
			if prefix_candidate.strip() in lex.affix_tokens:
				if lex.affix_tokens[prefix_candidate.strip()] == "prefix":
					return re.sub(r'^' + prefix_candidate, "", marktext)
	return marktext


def remove_infix_tokens(marktext, lex):
	"""
	Remove infix tokens such as dashes, interfixed articles (in Semitic construct state) etc.

	:param marktext: the markable text string to remove tokens from
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: potentially truncated text
	"""
	return lex.filters["core_infixes"].sub(" ", marktext)


def resolve_mark_entity(mark, lex):
	"""
	Main function to set entity type based on progressively less restricted parts of a markable's text

	:param mark: The :class:`.Markable` object to get the entity type for
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: void
	"""

	entity = ""
	use_entity_deps = True
	use_entity_sims = True
	use_sequencer = True if lex.sequencer is not None else False
	if "ablations" in lex.debug:
		if "no_entity_dep" in lex.debug["ablations"]:
			use_entity_deps = False
		if "no_entity_sim" in lex.debug["ablations"]:
			use_entity_sims= False
		if "no_sequencer" in lex.debug["ablations"]:
			use_sequencer = False

	## DEBUG POINT ##
	if mark.text == lex.debug["ana"] or mark.head.text == lex.debug["ana"]:
		a=5

	parent_text = mark.head.head_text
	if mark.form == "pronoun":
		if re.search(r'[12]',mark.agree):  # Explicit 1st or 2nd person pronoun
			entity = lex.filters["person_def_entity"]
			mark.entity_certainty = 'certain'
		elif mark.agree == "male" or mark.agree == "female":  # Possibly human 3rd person
			entity = lex.filters["person_def_entity"]
			mark.entity_certainty = 'uncertain'
		else:
			if use_sequencer:
				pred, score = mark.head.seq_pred
				if pred != "O":
					entity = pred
					mark.entity_certainty = 'sequencer'
			if use_entity_deps and entity == "":
				if parent_text in lex.entity_deps:
					if mark.head.func in lex.entity_deps[parent_text][mark.head.func]:
						dep_ents = dict(lex.entity_deps[parent_text][mark.head.func])
						if lex.filters["no_person_agree"].match(mark.agree) is not None and lex.filters["person_def_entity"] in dep_ents:
							del dep_ents[lex.filters["person_def_entity"]]
						if len(dep_ents) > 0:
							entity = max(iterkeys(dep_ents), key=(lambda key: dep_ents[key]))
				if entity == "":  # No literal match for dependency, fall back to similar heads
					if parent_text in lex.similar and use_entity_sims:
						similar_heads = lex.similar[parent_text]
						for similar_head in similar_heads:
							if similar_head in lex.entity_deps:
								if mark.head.func in lex.entity_deps[similar_head]:
									if lex.filters["no_person_agree"].match(mark.agree) is not None:
										similar_dict = {}
										for key, value in lex.entity_deps[similar_head][mark.head.func].items():
											if key != lex.filters["person_def_entity"]:
												similar_dict[key] = value
									else:
										similar_dict = lex.entity_deps[similar_head][mark.head.func]
									if len(similar_dict) > 0:
										entity = max(similar_dict,
													 key=(lambda key: similar_dict[key]))
										break
			if entity == "":  # Entity dependency information not used; no way to guess entity
				entity = lex.filters["default_entity"]
				mark.entity_certainty = "uncertain"
	else:
		if mark.coordinate:
			# For coordinate markables we expect the constituents to determine the entity in assign_coordinate_entity.
			# An exception to this is when the entire coordination is listed in the entities list.
			if entity == "":
				entity = resolve_entity_cascade(mark.text, mark, lex)
			if entity == "":
				entity = resolve_entity_cascade(mark.core_text, mark, lex)
		else:
			if entity == "":
				# Try to catch year numbers and hours + minutes
				if re.match(r'^(1[456789][0-9][0-9]|20[0-9][0-9]|(2[0-3]|1?[0-9]):[0-5][0-9]|ה?תש.".)$', mark.head.text) is not None:
					entity = lex.filters["time_def_entity"]
					mark.entity_certainty = "uncertain"
					mark.subclass = "time-unit" # TODO: de-hardwire this
					mark.definiteness = "def"  # literal year numbers are considered definite like 'proper names'
					mark.form = "proper"  # literal year numbers are considered definite like 'proper names'
			if entity == "":
				if re.match(r'^(([0-9]+[.,]?)+)$', mark.core_text) is not None:
					entity = lex.filters["quantity_def_entity"]
					mark.alt_entities.append(lex.filters["time_def_entity"])
					mark.entity_certainty = "uncertain"
			if entity == "":
				entity = resolve_entity_cascade(mark.text, mark, lex)
			if entity == "":
				entity = resolve_entity_cascade(replace_head_with_lemma(mark), mark, lex)
			if entity == "":
				entity = resolve_entity_cascade(remove_suffix_tokens(mark.text.strip(),lex), mark, lex)
			if entity == "":
				entity = resolve_entity_cascade(remove_prefix_tokens(mark.text.strip(), lex), mark, lex)
			if entity == "" and mark.core_text != mark.text:
				entity = resolve_entity_cascade(mark.core_text, mark, lex)
			if entity == "":
				entity = recognize_entity_by_mod(mark, lex)
			if entity == "" and mark.head.text.istitle():
				if mark.head.text in lex.last_names:
					modifiers_match_article = (lex.filters["articles"].match(mod.text) is not None for mod in mark.head.modifiers)
					modifiers_match_first_name = (mod.text in lex.first_names for mod in mark.head.modifiers)
					if any(modifiers_match_first_name) and not any(modifiers_match_article):
						entity = lex.filters["person_def_entity"]
			if entity == "" and mark.head.text.istitle():
				entity = resolve_entity_cascade(mark.core_text.lower(), mark, lex)
			if entity == "" and not mark.head.text.istitle():
				entity = resolve_entity_cascade(mark.core_text[:1].upper() + mark.core_text[1:], mark, lex)
			if entity == "":
				entity = resolve_entity_cascade(mark.head.text, mark, lex)
			if entity == "" and mark.head.text.istitle():
				entity = resolve_entity_cascade(mark.head.text.lower(), mark, lex)
			if entity == "" and mark.head.text.isupper():
				entity = resolve_entity_cascade(mark.head.text.lower(), mark, lex)
			if entity == "" and mark.head.text.isupper():
				entity = resolve_entity_cascade(mark.head.text.lower().title(), mark, lex)
			if entity == "" and not mark.head.lemma == mark.head.text:  # Try lemma match if lemma different from token
				entity = resolve_entity_cascade(mark.head.lemma, mark, lex)
			if entity == "":
				if (mark.head.text.istitle() or not lex.filters["cap_names"]):
					if mark.head.text in lex.last_names or mark.head.text in lex.first_names:
						modifiers_match_definite = (lex.filters["definite_articles"].match(mod.text) is not None for mod in mark.head.modifiers)
						modifiers_match_article = (lex.filters["articles"].match(mod.text) is not None for mod in mark.head.modifiers)
						modifiers_match_def_entity = (re.sub(r"\t.*","",lex.entity_heads[mod.text.strip().lower()][0]) == lex.filters["default_entity"] for mod in mark.head.modifiers if mod.text.strip().lower() in lex.entity_heads)
						if not (any(modifiers_match_article) or any(modifiers_match_definite) or any(modifiers_match_def_entity)):
							entity = lex.filters["person_def_entity"]
			if entity == "":
				# Just use sequencer if desired
				if use_sequencer:
					pred, score = mark.head.seq_pred
					if pred != "O":
						entity = pred
						mark.entity_certainty = 'sequencer'

			if entity == "":
				# See what the affix morphology predicts for the head
				head_text = mark.lemma if mark.lemma != "_" and mark.lemma != "" else mark.head.text
				morph_probs = get_entity_by_affix(head_text,lex)

				# Now check what the dependencies predict
				dep_probs = {}
				if use_entity_deps:
					if parent_text in lex.entity_deps:
						if mark.head.func in lex.entity_deps[parent_text]:
							dep_probs.update(lex.entity_deps[parent_text][mark.head.func])
					if len(dep_probs) == 0:  # No literal dependency information found, check if similar heads are known
						if parent_text in lex.similar:
							similar_heads = lex.similar[parent_text]
							for similar_head in similar_heads:
								if similar_head in lex.entity_deps:
									if mark.head.func in lex.entity_deps[similar_head]:
										dep_probs.update(lex.entity_deps[similar_head][mark.head.func])
										break

				# And check what entity similar words are
				sim_probs = {}
				if use_entity_sims:
					if mark.head.text in lex.similar:
						for similar_word in lex.similar[mark.head.text]:
							if similar_word in lex.entity_heads:
								for entity_type in lex.entity_heads[similar_word]:
									entity_string = entity_type.split("\t")[0]
									if entity_string in sim_probs:
										sim_probs[entity_string] += 1
									else:
										sim_probs.update({entity_string:1})

				# Compare scores to decide between affix vs. dependency evidence vs. embeddings
				dep_values = list(dep_probs[key] for key in dep_probs)
				total_deps = float(sum(dep_values))
				sim_values = list(sim_probs[key] for key in sim_probs)
				total_sims = float(sum(sim_values))
				norm_dep_probs = {}
				norm_sim_probs = {}

				# Normalize - each information source hedges its bets based on how many guesses it makes
				for key, value in iteritems(dep_probs):
					norm_dep_probs[key] = value/total_deps
				for key, value in iteritems(sim_probs):
					norm_sim_probs[key] = value/total_sims

				joint_probs = defaultdict(float)
				joint_probs.update(norm_dep_probs)
				for entity in morph_probs:
					joint_probs[entity] += morph_probs[entity]
				for entity in norm_sim_probs:
					joint_probs[entity] += sim_probs[entity]
				# Bias in favor of default entity to break ties
				joint_probs[lex.filters["default_entity"]] += 0.0000001

				entity = max(joint_probs, key=(lambda key: joint_probs[key]))

	if entity != "":
		mark.entity = entity

	if "/" in mark.entity:  # Lexicalized agreement information appended to entity
		if mark.agree == "" or mark.agree is None:
			mark.agree = mark.entity.split("/")[1]
		elif mark.agree_certainty == "":
			mark.alt_agree.append(mark.agree)
			mark.agree = mark.entity.split("/")[1]
		mark.entity = mark.entity.split("/")[0]
	elif mark.entity == lex.filters["person_def_entity"] and mark.agree == lex.filters["default_agree"] and mark.form != "pronoun":
		mark.agree = lex.filters["person_def_agree"]
		mark.agree_certainty = "uncertain"
	if "\t" in mark.entity:  # This is a subclass bearing solution
		mark.subclass = mark.entity.split("\t")[1]
		mark.entity = mark.entity.split("\t")[0]
	if mark.entity == lex.filters["person_def_entity"] and mark.form != "pronoun":
		if mark.text in lex.names:
			mark.agree = lex.names[mark.text]
	if mark.entity == lex.filters["person_def_entity"] and mark.agree is None:
		no_affix_mark = remove_suffix_tokens(remove_prefix_tokens(mark.text, lex), lex)
		if no_affix_mark in lex.names:
			mark.agree = lex.names[no_affix_mark]
	if mark.entity == lex.filters["person_def_entity"] and mark.agree is None:
		mark.agree = lex.filters["person_def_agree"]
		mark.agree_certainty = "uncertain"
	if mark.entity == "" and mark.core_text.upper() == mark.core_text and re.search(r"[A-ZÄÖÜ]", mark.core_text) is not None:  # Unknown all caps entity, guess acronym default
		mark.entity = lex.filters["all_caps_entity"]
		mark.entity_certainty = "uncertain"
	if mark.entity == "":  # Unknown entity, guess default
		mark.entity = lex.filters["default_entity"]
		mark.entity_certainty = "uncertain"
	if mark.subclass == "":
		if mark.subclass == "":
			mark.subclass = mark.entity
	if mark.func == "title":
		mark.entity = lex.filters["default_entity"]
	if mark.agree == "" and mark.entity == lex.filters["default_entity"]:
		mark.agree = lex.filters["default_agree"]


def resolve_entity_cascade(entity_text, mark, lex):
	"""
	Retrieve possible entity types for a given text fragment based on entities list, entity heads and names list.

	:param entity_text: The text to determine the entity for
	:param mark: The :class:`.Markable` hosting the text fragment to retrieve context information from (e.g. dependency)
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: entity type; note that this is used to decide whether to stop the search, but the Markable's entity is	already set during processing together with matching subclass and agree information
	"""
	options = {}
	entity = ""
	person_entity = lex.filters["person_def_entity"]
	if entity_text in lex.entities:
		for alt in lex.entities[entity_text]:
			parsed_entity = parse_entity(alt, "entities_match")
			if parsed_entity[0] not in mark.alt_entities:
				mark.alt_entities.append(parsed_entity[0])
				mark.alt_subclasses.append(parsed_entity[1])
				options[parsed_entity[0]] = parsed_entity
	if entity_text in lex.entity_heads:
		for alt in lex.entity_heads[entity_text]:
			parsed_entity = parse_entity(alt, "entity_heads_match")
			if parsed_entity[0] not in mark.alt_entities:
				mark.alt_entities.append(parsed_entity[0])
				mark.alt_subclasses.append(parsed_entity[1])
				options[parsed_entity[0]] = parsed_entity
	# Add the person entity based on a possible name despite seeing alternative entities
	# If and only if this is supported by a unique dependency cue (only person dependencies found, well attested)
	if entity_text in lex.names or entity_text in lex.last_names or entity_text in lex.first_names:
		if (entity_text[0].istitle() or not lex.filters["cap_names"]) and person_entity not in mark.alt_entities:
			if mark.head.head_text in lex.entity_deps:
				if mark.func in lex.entity_deps[mark.head.head_text]:
					if lex.filters["person_def_entity"] in lex.entity_deps[mark.head.head_text][mark.func]:
						# Must be attested > 5 times; relaxing this can lower precision substantially
						if lex.entity_deps[mark.head.head_text][mark.func][lex.filters["person_def_entity"]] > 5 and len(lex.entity_deps[mark.head.head_text][mark.func])==1:
							mark.alt_entities.append(lex.filters["person_def_entity"])
							mark.alt_subclasses.append(lex.filters["person_def_entity"])
							name_agree = ""
							if entity_text in lex.names:
								name_agree = lex.names[entity_text]
							elif entity_text in lex.first_names and not entity_text in lex.last_names:
								name_agree = lex.first_names[entity_text]
							options[person_entity] = (person_entity, person_entity, name_agree,"names_match")
	if len(mark.alt_entities) < 1 and 0 < entity_text.count(" ") < 3 and lex.filters["person_def_entity"] not in mark.alt_entities:
		if entity_text.split(" ")[0] in lex.first_names and entity_text.split(" ")[-1] in lex.last_names:
			if entity_text[0].istitle() or not lex.filters["cap_names"]:
				if lex.filters["articles"].match(mark.text.split(" ")[0]) is None:
					mark.alt_entities.append(person_entity)
					mark.alt_subclasses.append(person_entity)
					options[person_entity] = (person_entity, person_entity, lex.first_names[entity_text.split(" ")[0]], "name_match")

	if person_entity not in mark.alt_entities and (mark.text in lex.first_names or mark.text in lex.last_names):
		mark.alt_entities.append(person_entity)
		options[person_entity] = (person_entity,person_entity,'','name_match')
	if len(mark.alt_entities) > 1:
		entity = disambiguate_entity(mark, lex)
	elif len(mark.alt_entities) == 1:
		entity = mark.alt_entities[0]

	if entity != "":
		mark.entity, mark.subclass = options[entity][0:2]
		if options[entity][2] != "":
			mark.agree = options[entity][2]
		mark.entity_certainty = options[entity][3]

	return entity if len(options) > 0 else ""


def parse_entity(entity_text, certainty="uncertain"):
	"""
	Parses: entity -tab- subclass(/agree) + certainty into a tuple

	:param entity_text: the string to parse, must contain excatly two tabs
	:param certainty: the certainty string at end of tuple, default 'uncertain'
	:return: quadruple of (entity, subclass, agree, certainty)
	"""
	entity, subclass, freq = entity_text.split("\t")
	if "/" in subclass:
		subclass, agree = subclass.split("/")
	else:
		agree = ""
	return (entity, subclass, agree, certainty)


def resolve_mark_agree(mark, lex):
	"""
	Resolve Markable agreement based on morph information in tokens or gazetteer data

	:param mark: The :class:`.Markable` to resolve agreement for
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: void
	"""

	# DEBUG POINT #
	if mark.text == lex.debug["ana"]:
		a=5

	if mark.head.morph not in ["","_"]:
		mark.agree_certainty = "head_morph"
		return [mark.head.morph]
	else:
		if mark.form == "pronoun":
			if mark.text in lex.pronouns:
				return lex.pronouns[mark.text]
			elif mark.text.lower() in lex.pronouns:
				return lex.pronouns[mark.text.lower()]
		if mark.form == "proper":
			if mark.core_text in lex.names:
				return [lex.names[mark.core_text]]
			elif mark.core_text in lex.first_names and mark.core_text not in lex.entities and mark.core_text not in lex.entity_heads:  # Single name component core text
				return [lex.first_names[mark.core_text]]
		if mark.head.pos in lex.pos_agree_mappings:
			mark.agree_certainty = "pos_agree_mappings"
			return [lex.pos_agree_mappings[mark.head.pos]]
		elif mark.core_text in lex.entities:
			for full_entry in lex.entities[mark.core_text]:
				entry = full_entry.split("\t")[1]
				if "/" in entry:
					if mark.agree == "":
						mark.agree = entry[entry.find("/") + 1:]
					mark.alt_agree.append(entry[entry.find("/") + 1:])
		elif mark.head.text in lex.entity_heads:
			for full_entry in lex.entity_heads[mark.head.text]:
				entry = full_entry.split("\t")[1]
				if "/" in entry:
					if mark.agree == "":
						mark.agree = entry[entry.find("/") + 1:]
					mark.alt_agree.append(entry[entry.find("/") + 1:])


def resolve_cardinality(mark,lex):
	"""
	Find cardinality for Markable based on numerical modifiers or number words

	:param mark: The :class:`.Markable` to resolve agreement for
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: Cardinality as float, zero if unknown
	"""
	def check_card(mod):
		if mod in lex.numbers:
			return int(lex.numbers[mod][0])
		elif mod.lower() in lex.numbers:
			return int(lex.numbers[mod.lower()][0])
		else:
			thousand_sep = r"\." if lex.filters["thousand_sep"] == "." else lex.filters["thousand_sep"]
			pure_number_candidate = re.sub(thousand_sep,"",mod)

			decimal_sep = lex.filters["decimal_sep"]
			if decimal_sep != ".":
				pure_number_candidate = re.sub(decimal_sep,".",pure_number_candidate)
			if re.match("^(\d+(\.\d+)?|(\.\d+))$",pure_number_candidate) is not None:
				return float(pure_number_candidate)
			else:
				parts = re.match("^(\d+)/(\d+)$",pure_number_candidate)
				if parts is not None: # Fraction with slash division
					numerator = float(parts.groups()[0])
					denominator = float(parts.groups()[1])
					return numerator/denominator

	for m in mark.head.modifiers:
		card = check_card(m.text)
		if card is not None:
			return card
	card = check_card(mark.head.text)
	if card is not None:
		return card
	card = check_card(mark.head.lemma)
	if card is not None:
		return card

	return 0


def recognize_entity_by_mod(mark, lex, mark_atoms=False):
	"""
	Attempt to recognize entity type based on modifiers
	
	:param mark: :class:`.Markable` for which to identify the entity type
	:param modifier_lexicon: The :class:`.LexData` object's modifier list
	:return: String (entity type, possibly including subtype and agreement)
	"""
	modifier_lexicon = lex.entity_mods
	mod_atoms = lex.mod_atoms
	for mod in mark.head.modifiers:
		modifier_tokens = [mod.text] + [construct_modifier_substring(mod)]
		while len(modifier_tokens) > 0:
			identifying_substr = ""
			for token in modifier_tokens:
				identifying_substr += token + " "
				if identifying_substr.strip() in modifier_lexicon:
					if identifying_substr.strip() in mod_atoms and mark_atoms:
						return modifier_lexicon[identifying_substr.strip()][0] + "@"
					else:
						return modifier_lexicon[identifying_substr.strip()][0]
				elif identifying_substr.lower().strip() in modifier_lexicon:
					if identifying_substr.lower().strip() in mod_atoms and mark_atoms:
						return modifier_lexicon[identifying_substr.lower().strip()][0] + "@"
					else:
						return modifier_lexicon[identifying_substr.lower().strip()][0]
			modifier_tokens.pop(0)
	return ""


def construct_modifier_substring(modifier):
	"""
	Creates a list of tokens representing a modifier and all of its submodifiers in sequence
	
	:param modifier: A ParsedToken object from the modifier list of the head of some markable
	:return: Text of that modifier together with its modifiers in sequence
	"""
	candidate_prefix = ""
	mod_dict = get_mod_ordered_dict(modifier)
	for mod_member in mod_dict:
		candidate_prefix += mod_dict[mod_member].text + " "
	return candidate_prefix.strip()


def stoplist_prefix_tokens(mark, prefix_dict, keys_to_pop):
	substr = ""
	candidate_prefix = ""
	for mod in mark.head.modifiers:
		mod_dict = get_mod_ordered_dict(mod)
		for mod_member in mod_dict:
			candidate_prefix += mod_dict[mod_member].text + " "
		tokens = candidate_prefix.strip().split(" ")
		for token in tokens:
			substr += token + " "
			if substr.strip() in prefix_dict:
				tokens_affected_count = substr.count(" ")
				i = 0
				for mod_token in mod_dict:
					if i < tokens_affected_count and not mod_dict[mod_token].id == mark.head.id:
						keys_to_pop.append(mod_dict[mod_token].id)
					i += 1


def get_mod_ordered_dict(mod):
	"""
	Retrieves the (sub)modifiers of a modifier token
	
	:param mod: A :class:`.ParsedToken` object representing a modifier of the head of some markable
	:return: Recursive ordered dictionary of that modifier's own modifiers
	"""
	mod_dict = OrderedDict()
	mod_dict[int(mod.id)] = mod
	if len(mod.modifiers) > 0:
		for mod2 in mod.modifiers:
			mod_dict.update(get_mod_ordered_dict(mod2))
	else:
		return mod_dict
	return OrderedDict(sorted(mod_dict.items()))


def markable_extend_punctuation(marktext, adjacent_token, punct_dict, direction):
	if direction == "trailing":
		for open_punct in punct_dict:
			if (" " + open_punct + " " in marktext or marktext.startswith(open_punct + " ")) and adjacent_token.text == punct_dict[open_punct]:
				return True
	else:
		for close_punct in punct_dict:
			if (" " + close_punct + " " in marktext or marktext.endswith(" " + close_punct)) and adjacent_token.text == punct_dict[close_punct]:
				return True
	return False


def markables_overlap(mark1, mark2, lex=None):
	"""
	Helper function to check if two markables cover some of the same tokens. Note that if the lex argument is specified,
	it is used to recognize possessives, which behave exceptionally. Possessive pronouns beginning
	after a main markable has started are tolerated in case of markable definitions including relative clauses,
	e.g. [Mr. Pickwick, who was looking for [his] hat]

	:param mark1: First :class:`.Markable`
	:param mark2: Second :class:`.Markable`
	:param lex: the :class:`.LexData` object with gazetteer information and model settings or None
	:return: bool
	"""
	if lex is not None:
		if lex.filters["possessive_func"].match(mark1.func) is not None and mark1.form == "pronoun" and mark1.start > mark2.start:
			return False
		elif lex.filters["possessive_func"].match(mark2.func) is not None and mark2.form == "pronoun" and mark2.start > mark1.start:
			return False
	if mark2.end >= mark1.start >= mark2.start:
		return True
	elif mark2.end >= mark1.end >= mark2.start:
		return True
	else:
		return False


def markable_extend_affixes(start, end, conll_tokens, sent_start, lex):
	candidate_affix = ""
	for tok in reversed(conll_tokens[sent_start:start]):
		candidate_affix = tok.text + " " + candidate_affix
		if candidate_affix.lower().strip() in lex.affix_tokens:
			if lex.affix_tokens[candidate_affix.lower().strip()] == "prefix":
				return [int(tok.id), int(tok.id) + candidate_affix.count(" ")]
		elif candidate_affix.strip() in lex.affix_tokens:
			if lex.affix_tokens[candidate_affix.strip()] == "prefix":
				return [int(tok.id), int(tok.id) + candidate_affix.count(" ")]
	candidate_affix = ""
	for tok in conll_tokens[end+1:]:
		candidate_affix += tok.text + " "
		if candidate_affix.lower().strip() in lex.affix_tokens:
			if lex.affix_tokens[candidate_affix.lower().strip()] == "suffix":
				return [int(tok.id) - candidate_affix.strip().count(" "), int(tok.id) + 1]
		elif candidate_affix.strip() in lex.affix_tokens:
			if lex.affix_tokens[candidate_affix.strip()] == "suffix":
				return [int(tok.id) - candidate_affix.strip().count(" "), int(tok.id) + 1]
	return [0,0]


def get_entity_by_affix(head_text, lex):
	affix_max = 0
	score = 0
	entity = ""
	probs = {}
	for i in range(1, len(head_text) - 1):
		candidates = 0
		if head_text[i:] in lex.morph:
			options = lex.morph[head_text[i:]]
			for key, value in options.items():
				candidates += value
				entity = key.split("/")[0]
				probs[entity] = float(value)
			for entity in probs:
				probs[entity] = probs[entity] / candidates
		if entity != "":
			return probs
	return probs


def pos_func_combo(pos, func, pos_func_heads_string):
	"""
	:return: bool
	"""
	pos_func_heads = pos_func_heads_string.split(";")
	if pos + "+" + func in pos_func_heads:
		return True
	elif pos + "!" + func in pos_func_heads:
		return False
	else:
		if pos_func_heads_string.find(";" + pos + "!") > -1 or pos_func_heads_string.startswith(pos + "!"):
			return True
		else:
			return False


def replace_head_with_lemma(mark):
	head = re.escape(mark.head.text)
	lemma = mark.head.lemma
	return re.sub(head, lemma, mark.core_text).strip()


def make_markable(tok, conll_tokens, descendants, tokoffset, sentence, keys_to_pop, lex):
	if tok.id in descendants and lex.filters["non_extend_pos"].match(tok.pos) is None:
		tokenspan = descendants[tok.id] + [tok.id]
		tokenspan = list(map(int, tokenspan))
		tokenspan.sort()
		marktext = ""
		start = min(tokenspan)
		end = max(tokenspan)
		for span_token in conll_tokens[start:end + 1]:
			marktext += span_token.text + " "
		marktext = marktext.strip()
	else:
		marktext = tok.text
		start = int(tok.id)
		end = int(tok.id)
	# Check for a trailing coordinating conjunction on a descendant of the head and re-connect if necessary
	if end < len(conll_tokens) - 1:
		coord = conll_tokens[end + 1]
		if lex.filters["cc_left_to_right"]:
			not_head_child = coord.head != tok.id
		else:
			coord_grand_head = 0
			if int(conll_tokens[int(coord.head)].head) != 0:
				coord_grand_head = int(conll_tokens[int(coord.head)].head)
			not_head_child = (conll_tokens[int(coord.head)].head != tok.id
									and coord_grand_head == int(tok.id)
									and conll_tokens[int(coord.head)].head != '0'
									and int(conll_tokens[int(coord.head)].head) > int(tok.id))

		if lex.filters["coord_func"].match(coord.func) is not None and not_head_child and int(coord.head) >= start:
			conjunct1 = conll_tokens[int(conll_tokens[end + 1].head)]
			for tok2 in conll_tokens[end + 1:]:
				if (tok2.head == conjunct1.head and tok2.func == conjunct1.func) or tok2.head == coord.id:
					conjunct2 = tok2
					tokenspan = [conjunct2.id, str(end)]
					if conjunct2.id in descendants:
						tokenspan += descendants[conjunct2.id]
					tokenspan = map(int, tokenspan)
					tokenspan = sorted(tokenspan)
					end = max(tokenspan)
					marktext = ""
					for span_token in conll_tokens[start:end + 1]:
						marktext += span_token.text + " "
					break

	core_text = marktext.strip()
	# DEBUG POINT
	if marktext.strip() in lex.debug:
		pass
	# Extend markable to 'affix tokens'
	# Do not extend pronouns or stop functions
	if lex.filters["stop_func"].match(tok.func) is None and lex.filters["pronoun_pos"].match(tok.pos) is None:
		extend_affixes = markable_extend_affixes(start, end, conll_tokens, tokoffset + 1, lex)
		if not extend_affixes[0] == 0:
			if extend_affixes[0] < start:
				prefix_text = ""
				for prefix_tok in conll_tokens[extend_affixes[0]:extend_affixes[1]]:
					prefix_text += prefix_tok.text + " "
					keys_to_pop.append(prefix_tok.id)
					start -= 1
				marktext = prefix_text + marktext
			else:
				for suffix_tok in conll_tokens[extend_affixes[0]:extend_affixes[1]]:
					keys_to_pop.append(suffix_tok.id)
					marktext += suffix_tok.text + " "
					end += 1

	# Extend markable to trailing closing punctuation if it contains opening punctuation
	if end < len(conll_tokens) - 1:
		next_id = end + 1
		if markable_extend_punctuation(marktext, conll_tokens[next_id], lex.open_close_punct, "trailing"):
			marktext += conll_tokens[next_id].text + " "
			end += 1
	if start != "1":
		prev_id = start - 1
		if markable_extend_punctuation(marktext, conll_tokens[prev_id], lex.open_close_punct_rev, "leading"):
			marktext = conll_tokens[prev_id].text + " " + marktext
			start -= 1

	this_markable = Markable("", tok, "", "", start, end, core_text, core_text, "", "", "", "new", "", sentence, "none", "none", 0, [], [], [])
	# DEBUG POINT
	if this_markable.text == lex.debug["ana"]:
		pass
	this_markable.core_text = remove_infix_tokens(remove_suffix_tokens(remove_prefix_tokens(this_markable.core_text,lex),lex),lex)
	while this_markable.core_text != core_text:  # Potentially repeat affix stripping as long as core text changes
		core_text = this_markable.core_text
		this_markable.core_text = remove_infix_tokens(remove_suffix_tokens(remove_prefix_tokens(this_markable.core_text,lex),lex),lex)
	if this_markable.core_text == '':  # Check in case suffix removal has left no core_text
		this_markable.core_text = marktext.strip()
	this_markable.text = marktext  # Update core_text with potentially modified markable text
	return this_markable


def lookup_has_entity(text, lemma, entity, lex):
	"""
	Checks if a certain token text or lemma have the specific entity listed in the entities or entity_heads lists
	
	:param text: text of the token
	:param lemma: lemma of the token
	:param entity: entity to check for
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: bool
	"""
	found = []
	if text in lex.entities:
		found = [i for i, x in enumerate(lex.entities[text]) if re.search(entity + '\t', x)]
	elif lemma in lex.entities:
		found = [i for i, x in enumerate(lex.entities[lemma]) if re.search(entity + '\t', x)]
	elif text in lex.entity_heads:
		found = [i for i, x in enumerate(lex.entity_heads[text]) if re.search(entity + '\t', x)]
	elif lemma in lex.entity_heads:
		found = [i for i, x in enumerate(lex.entity_heads[lemma]) if re.search(entity + '\t', x)]
	return len(found) > 0


def assign_coordinate_entity(mark,markables_by_head):
	"""
	Checks if all constituents of a coordinate markable have the same entity and subclass and if so, propagates these to the coordinate markable.

	:param mark: a coordinate markable to check the entities of its constituents
	:param markables_by_head: dictionary of markables by head id
	:return: void
	"""

	sub_entities = []
	sub_subclasses = []
	for m_id in mark.submarks:
		if m_id in markables_by_head:
			sub_entities.append(markables_by_head[m_id].entity)
			sub_subclasses.append(markables_by_head[m_id].subclass)
	if len(set(sub_entities)) == 1:  # There is agreement on the entity
		mark.entity = sub_entities[0]
	if len(set(sub_subclasses)) == 1:  # There is agreement on the entity
		mark.subclass = sub_subclasses[0]


def disambiguate_entity(mark,lex):
	"""
	Selects prefered entity for a Markable with multiple alt_entities based on dependency information or more common type

	:param mark: the Markable object
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: predicted entity type as string
	"""
	##DEBUG POINT##
	if mark.text in lex.debug["ana"]:
		a=3

	# Prefer sequencer entity if it is one of the options
	use_sequencer = True if lex.sequencer is not None else False
	if use_sequencer:
		seq_ent = mark.head.seq_pred[0]
		if seq_ent in mark.alt_entities:
			return seq_ent

	parent_text = mark.head.head_text
	scores = defaultdict(float)
	entity_freqs = defaultdict(int)
	# Bias tie breaker in favor of default entity
	if lex.filters["default_entity"] in mark.alt_entities:
		scores[lex.filters["default_entity"]] += 0.0001
	if parent_text in lex.entity_deps:
		if mark.func in lex.entity_deps[parent_text]:
			for alt_entity in mark.alt_entities:
				if alt_entity in lex.entity_deps[parent_text][mark.func]:
					entity_freqs[alt_entity] = lex.entity_deps[parent_text][mark.func][alt_entity]

	if len(entity_freqs) == 0:  # No dependency info, use similar dependencies if available
		if parent_text in lex.similar:
			for similar_parent in lex.similar[parent_text]:
				if similar_parent in lex.entity_deps:
					if mark.func in lex.entity_deps[similar_parent]:
						for alt_entity in mark.alt_entities:
							if alt_entity in lex.entity_deps[similar_parent][mark.func]:
								entity_freqs[alt_entity] = lex.entity_deps[similar_parent][mark.func][alt_entity]

	# Check for ties
	break_tie = False
	if len(entity_freqs) > 0:
		best_freq = max(entity_freqs.values())
		best_entities = [k for k, v in entity_freqs.items() if v == best_freq]
		if len(best_entities) > 1:
			break_tie = True

	if len(entity_freqs) == 0 or break_tie:  # No similar dependencies, get frequency information from entities if available
		if mark.text in lex.entities:
			for entity_entry in lex.entities[mark.text]:
				entity_type, entity_subtype, freq = entity_entry.split("\t")
				freq = int(freq)
				if freq > 0:
					entity_freqs[entity_type] += freq
	if len(entity_freqs) == 0 or break_tie:  # No similar dependencies, get frequency information from heads if available
		if mark.head.text in lex.entity_heads:
			for entity_entry in lex.entity_heads[mark.head.text]:
				entity_type, entity_subtype, freq = entity_entry.split("\t")
				freq = int(freq)
				if freq > 0:
					entity_freqs[entity_type] += freq

	if len(entity_freqs) == 0:  # No dependency info, use entity sum proportions
		entity_freqs = lex.entity_sums

	for entity_type in mark.alt_entities:
		scores[entity_type] += entity_freqs[entity_type]


	best_entity = max(iterkeys(scores), key=(lambda key: scores[key]))
	return best_entity

