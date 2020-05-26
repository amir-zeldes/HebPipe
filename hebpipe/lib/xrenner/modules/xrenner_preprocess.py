"""
modules/xrenner_preprocess.py

Prepare parser output for entity and coreference resolution

Author: Amir Zeldes
"""

from .xrenner_marker import lookup_has_entity

def add_negated_parents(conll_tokens, offset):
	"""
	Sets the neg_parent property on tokens whose head dominates a negation

	:param conll_tokens: token list for this document
	:param offset: token ID reached in last sentence
	:return: None
	"""

	for token in conll_tokens[offset:]:
		parent_id = token.head
		if parent_id != "0":
			if conll_tokens[int(parent_id)].negated:
				token.neg_parent = True


def add_child_info(conll_tokens, child_funcs, child_strings, lex):
	"""
	Adds a list of all dependent functions and token strings to each parent token
	
	:param conll_tokens: The ParsedToken list so far
	:param child_funcs: Dictionary from ids to child functions
	:param child_strings: Dictionary from ids to child strings
	:return: void
	"""
	for child_id in child_funcs:
		if child_id > len(conll_tokens)-1:
			continue
		for func in child_funcs[child_id]:
			if func not in conll_tokens[child_id].child_funcs:
				conll_tokens[child_id].child_funcs.append(func)
				if lex.filters["neg_func"].match(func):
					conll_tokens[child_id].negated = True
		for tok_text in child_strings[child_id]:
			if tok_text not in conll_tokens[child_id].child_strings:
				conll_tokens[child_id].child_strings.append(tok_text)


def postprocess_parser(conll_tokens, tokoffset, children, stop_ids, lex):
	for tok1 in conll_tokens[tokoffset + 1:]:
		if tok1.text == "-LSB-" or tok1.text == "-RSB-":
			tok1.pos = tok1.text
			tok1.func = "punct"
			tok1.head = "0"
		if lex.filters["mark_head_pos"].match(tok1.pos) is not None:
			entity_candidate = tok1.text + " "
			for tok2 in conll_tokens[int(tok1.id) + 1:]:
				if lex.filters["mark_head_pos"].match(tok2.pos) is not None:
					entity_candidate += tok2.text + " "
					### DEBUG BREAKPOINT ###
					if entity_candidate.strip() == lex.debug["ana"]:
						pass
					if entity_candidate.strip() in lex.entities:  # Entity matched, check if all tokens are inter-connected
						for tok3 in conll_tokens[int(tok1.id):int(tok2.id)]:
							# Ensure right most token has head outside entity:
							if int(tok2.head) > int(tok2.id) or int(tok2.head) < int(tok1.id):
								if (int(tok3.head) < int(tok1.id) or int(tok3.head) > int(tok2.id)) and tok3.id in children[tok3.head]:
									children[tok3.head].remove(tok3.id)
									tok3.head = tok2.id
									children[tok3.head].append(tok3.id)
									break
				else:
					break
		# Check for apposition pointing back to immediately preceding proper noun token -
		# typical (German model) MaltParser name behavior
		if lex.filters["apposition_func"].match(tok1.func) is not None and not tok1.id == "1":
			if lex.filters["proper_pos"].match(conll_tokens[int(tok1.id) - 1].pos) is not None and conll_tokens[
						int(tok1.id) - 1].id == tok1.head:
				tok1.func = "xrenner_fix"
				children[str(int(tok1.id) - 1)].append(tok1.id)
				stop_ids[tok1.id] = True

		# Check for [city], [state/country] apposition -
		# typical (English model) Stanford parser behavior
		if tok1.text == lex.debug["ana"]:
			a=5
		if lex.filters["apposition_func"].match(tok1.func) is not None and not int(tok1.id) < 3:
			if conll_tokens[int(tok1.id) - 1].text.strip() == ",":
				tok_minus2 = conll_tokens[int(tok1.id) - 2]
				tok1_head = conll_tokens[int(tok1.head)]
				if lex.filters["proper_pos"].match(tok_minus2.pos) is not None:
					if (tok_minus2.id == tok1.head and (lookup_has_entity(tok1.text, tok1.lemma, "place", lex) and not lookup_has_entity(tok_minus2.text, tok_minus2.lemma, "place", lex) or \
						lookup_has_entity(tok_minus2.text, tok_minus2.lemma, "place", lex))) or \
						not lookup_has_entity(tok1_head.text, tok1_head.lemma, "place", lex) and lookup_has_entity(tok1.text, tok1.lemma, "place", lex):
							tok1.func = "xrenner_fix"
							if tok1.id not in children[tok_minus2.id]:
								if tok_minus2.head != tok1.id:  # Avoid creating a cycle
									children[tok_minus2.id].append(tok1.id)

		# Check for markable projecting beyond an apposition to itself and remove from children on violation
		if lex.filters["apposition_func"].match(tok1.func) is not None and not tok1.id == "1":
			for tok2 in conll_tokens[int(tok1.id) + 1:]:
				if tok2.head == tok1.head and lex.filters["non_link_func"].match(tok2.func) is None and tok2.id in children[tok2.head]:
					children[tok2.head].remove(tok2.id)


def replace_conj_func(conll_tokens, tokoffset, lex):
	"""
	Function to replace functions of tokens matching the conjunction function with their parent's function
	
	:param conll_tokens: The ParsedToken list so far
	:param tokoffset: The starting token for this sentence
	:param lex: the LexData object with gazetteer information and model settings
	:return: void
	"""

	for token in conll_tokens[tokoffset:]:
		## DEBUG POINT ##
		if token.text == lex.debug["ana"]:
			pass

		if lex.filters["conjunct_func"].match(token.func) is not None:
			for child_func in conll_tokens[int(token.head)].child_funcs:
				token.child_funcs.append(child_func)
			token.func = conll_tokens[int(token.head)].func
			token.head = conll_tokens[int(token.head)].head
			token.coordinate = True
