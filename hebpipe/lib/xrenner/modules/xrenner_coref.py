from .xrenner_marker import *
from .xrenner_compatible import *
from .xrenner_propagate import *
from .xrenner_rule import CorefRule, ConstraintMatcher

"""
Coreference resolution module. Iterates through markables to find possible matches based on rules.

Author: Amir Zeldes
"""


def find_antecedent(markable, previous_markables, lex, restrict_rule=""):
	"""
	Search for antecedents by cycling through coref rules for previous markables
	
	:param markable: Markable object to find an antecedent for
	:param previous_markables: Markables in all sentences up to and including current sentence
	:param lex: the LexData object with gazetteer information and model settings
	:param restrict_rule: a string specifying a subset of rules that should be checked (e.g. only rules with 'appos')
	:return: candidate, matching_rule - the best antecedent and the rule that matched it
	"""

	# DEBUG point
	if markable.text == lex.debug["ana"]:
		a=5
	candidate = None
	matching_rule = None
	for i, rule in enumerate(lex.coref_rules):
		# If this call of find_antecedent is limited to certain rules, check that the restriction is in the rule
		if restrict_rule == "" or restrict_rule in rule.ana_spec:
			if coref_rule_applies(lex, rule.ana_constraints, markable):
				candidate = search_prev_markables(markable, previous_markables, rule, lex)
				if candidate is not None:
					matching_rule = rule.propagation
					break

	return candidate, matching_rule


def search_prev_markables(markable, previous_markables, rule, lex):
	"""
	Search for antecedent to specified markable using a specified rule
	
	:param markable: The markable object to find an antecedent for
	:param previous_markables: The list of know markables up to and including the current sentence; markables beyond current markable but in its sentence are included for cataphora.
	:param ante_constraints: A list of ContraintMatcher objects describing the antecedent
	:param ante_spec: The antecedent specification part of the coref rule being checked, as a string
	:param lex: the LexData object with gazetteer information and model settings
	:param max_dist: Maximum distance in sentences for the antecedent search (0 for search within sentence)
	:param propagate: Whether to progpagate features upon match and in which direction
	:return: the selected candidate Markable object
	"""

	ante_constraints, ante_spec, rule_num, max_dist, propagate, clf_name = rule.ante_constraints, rule.ante_spec, rule.rule_num, rule.max_distance, rule.propagation, rule.clf_name

	candidate_set = set([])
	if ante_spec.find("lookahead") > -1:
		referents_to_loop = previous_markables
	else:
		referents_to_loop = reversed(previous_markables)
	for candidate in referents_to_loop:  # loop through previous markables backwards

		#DEBUG breakpoint:
		if markable.text == lex.debug["ana"]:
			a = 5
			if candidate.text == lex.debug["ante"]:
				b=6
		if markable.sentence.sent_num - candidate.sentence.sent_num <= max_dist:
			if ((int(markable.head.id) > int(candidate.head.id) and
			ante_spec.find("lookahead") == -1) or (int(markable.head.id) < int(candidate.head.id) and ante_spec.find("lookahead") > -1)):
				if candidate.group not in markable.non_antecdent_groups:
					if coref_rule_applies(lex, ante_constraints, candidate, markable):
						if not markables_overlap(markable, candidate, lex):
							if markable.form == "pronoun":
								if agree_compatible(markable, candidate, lex) or (ante_spec.find("anyagree") > -1 and group_agree_compatible(markable,candidate,previous_markables,lex)):
									if entities_compatible(markable, candidate, lex) and cardinality_compatible(markable, candidate, lex):
										candidate_set.add(candidate)
							elif markable.text == candidate.text or (len(markable.text) > 4 and (candidate.text.lower() == markable.text.lower())):
								#propagate_entity(markable, candidate, propagate)
								candidate_set.add(candidate)
								#return candidate
							elif markable.text + "|" + candidate.text in lex.coref and entities_compatible(
									markable, candidate, lex) and agree_compatible(markable, candidate, lex):
								candidate_set.add(candidate)
								#return candidate
							elif markable.core_text + "|" + candidate.core_text in lex.coref and entities_compatible(
									markable, candidate, lex) and agree_compatible(markable, candidate, lex):
								candidate_set.add(candidate)
								#return candidate
							elif markable.entity == candidate.entity and agree_compatible(markable, candidate, lex) and (markable.head.text == candidate.head.text or
							(len(markable.head.text) > 3 and (candidate.head.text.lower() == markable.head.text.lower())) or
							(markable.core_text.count(" ") > 2 and (markable.core_text.lower() == candidate.core_text.lower())) or
							(markable.head.lemma == candidate.head.lemma and lex.filters["lemma_match_pos"].match(markable.head.pos) is not None
							and lex.filters["lemma_match_pos"].match(candidate.head.pos) is not None)):
								if modifiers_compatible(markable, candidate, lex) and modifiers_compatible(candidate, markable, lex):
									candidate_set.add(candidate)
							elif (markable.entity == candidate.entity or len(set(markable.alt_entities) & set(candidate.alt_entities))>0) and isa(markable, candidate, lex):
								candidate.isa = True  # This is an 'isa' candidate
								candidate_set.add(candidate)
							elif agree_compatible(markable,candidate,lex) and ((markable.head.text == candidate.head.text) or (markable.head.lemma == candidate.head.lemma and
							lex.filters["lemma_match_pos"].match(markable.head.pos) is not None and lex.filters["lemma_match_pos"].match(candidate.head.pos) is not None)):
								if merge_entities(markable, candidate, previous_markables, lex):
									candidate_set.add(candidate)
							elif entities_compatible(markable, candidate, lex) and isa(markable, candidate, lex):
								if merge_entities(markable, candidate, previous_markables, lex):
									candidate.isa = True  # This is an 'isa' candidate
									candidate_set.add(candidate)
						elif lex.filters["match_acronyms"] and markable.head.text.isupper() or candidate.head.text.isupper():
								if acronym_match(markable, candidate, lex) or acronym_match(candidate, markable, lex):
									if modifiers_compatible(markable, candidate, lex) and modifiers_compatible(candidate, markable, lex):
										if merge_entities(markable, candidate, previous_markables, lex):
											candidate_set.add(candidate)
						if ante_spec.find("anytext") > -1:
								if (ante_spec.find("anyagree") > -1 and group_agree_compatible(markable,candidate,previous_markables,lex)) or agree_compatible(markable, candidate, lex):
									if (ante_spec.find("anycardinality") > -1 or cardinality_compatible(markable,candidate,lex)):
										if (ante_spec.find("anyentity") > -1 or entities_compatible(markable,candidate,lex)):
											candidate_set.add(candidate)
		elif ante_spec.find("lookahead") == -1:
			# Reached back too far according to max_dist, stop looking
			break

	if len(candidate_set) > 0:
		candidates_to_remove = set([])
		for candidate_item in candidate_set:
			# Remove items that are prohibited by entity agree mapping
			for agree, ent in iteritems(lex.filters["agree_entity_mapping"]):
				if markable.agree == agree and candidate_item.entity != ent:
					candidates_to_remove.add(candidate_item)
			if candidate_item.entity == lex.filters["person_def_entity"] and (candidate_item.form != "pronoun" or markable.entity_certainty == "certain") and lex.filters["no_person_agree"].match(markable.agree) is not None:
				candidates_to_remove.add(candidate_item)
			elif markable.entity == lex.filters["person_def_entity"] and (markable.form != "pronoun" or markable.entity_certainty == "certain") and lex.filters["no_person_agree"].match(candidate_item.agree) is not None:
				candidates_to_remove.add(candidate_item)

		for removal in candidates_to_remove:
			candidate_set.remove(removal)

		if len(candidate_set) > 0:
			take_first = True if ante_spec.find("takefirst") > -1 else False
			best = best_candidate(markable, candidate_set, lex, rule, take_first=take_first)
			if best is not None:
				if markable.text + "|" + best.text in lex.coref:
					markable.coref_type = lex.coref[markable.text + "|" + best.text]
					propagate_entity(markable, best, propagate)
					propagate_entity(markable, best)
				elif markable.core_text + "|" + best.core_text in lex.coref:
					markable.coref_type = lex.coref[markable.core_text + "|" + best.core_text]
					propagate_entity(markable, candidate_item)
				elif propagate.startswith("propagate"):
					propagate_entity(markable, best, propagate)
			if hasattr(best,"isa"):
				if hasattr(best,"isa_dir"):
					if best.isa_dir == "markable":
						markable.isa_partner_head = best.lemma
					else:
						best.isa_partner_head = markable.lemma
					delattr(best,"isa_dir")
				delattr(best,"isa")
			return best
		else:
			return None
	else:
		return None


def coref_rule_applies(lex, constraints, mark, anaphor=None):
	"""
	Check whether a markable definition from a coref rule applies to this markable
	
	:param lex: the LexData object with gazetteer information and model settings
	:param constraints: the constraints defining the relevant Markable
	:param mark: the Markable object to check constraints against
	:param anaphor: if this is an antecedent check, the anaphor is passed for $1-style constraint checks
	:return: bool: True if 'mark' fits all constraints, False if any of them fail
	"""
	for constraint in constraints:
		if not constraint.match(mark,lex,anaphor):
			return False
	return True


def antecedent_prohibited(markable, conll_tokens, lex):
	"""
	Check whether a Markable object is prohibited from having an antecedent
	
	:param markable: The Markable object to check
	:param conll_tokens: The list of ParsedToken objects up to and including the current sentence
	:param lex: the LexData object with gazetteer information and model settings
	:return: bool
	"""
	mismatch = True
	if "/" in lex.filters["no_antecedent"]:
		constraints = lex.filters["no_antecedent"].split(";")
		for constraint in constraints:
			if not mismatch:
				return True
			descriptions = constraint.split("&")
			mismatch = False
			for token_description in descriptions:
				if token_description.startswith("^"):
					test_token = conll_tokens[markable.start]
				elif token_description.startswith("$"):
					test_token = conll_tokens[markable.end]
				elif token_description.startswith("@"):
					test_token = markable.head
				else:
					# Invalid token description
					return False
				token_description = token_description[1:]
				pos, word = token_description.split("/")
				if pos.startswith("!"):
					pos = pos[1:]
					negative_pos = True
				else:
					negative_pos = False
				if word.startswith("!"):
					word = word[1:]
					negative_word = True
				else:
					negative_word = False
				pos_matcher = re.compile(pos)
				word_matcher = re.compile(word)
				if (pos_matcher.match(test_token.pos) is None and not negative_pos) or (pos_matcher.match(test_token.pos) is not None and negative_pos) or \
				(word_matcher.match(test_token.text) is None and not negative_word) or (word_matcher.match(test_token.text) is not None and negative_word):
					mismatch = True
					break
	if mismatch:
		return False
	else:
		return True



