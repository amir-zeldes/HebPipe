"""
Postprocessing module. Alters results of coreference analysis based on model settings,
such as deleting certain markables or re-wiring coreference relations according to a particular
annotation scheme

Author: Amir Zeldes and Shuo Zhang
"""

from collections import defaultdict
from .xrenner_classes import *
from .xrenner_marker import markables_overlap, markable_extend_punctuation


def postprocess_coref(markables, lex, markstart, markend, markbyhead, conll_tokens):


	# Collect markable groups
	marks_by_group = defaultdict(list)
	for markable in markables:
		if markable.antecedent != "none":
			if markable.group != markable.antecedent.group:
				markable.group = markable.antecedent.group
		marks_by_group[markable.group].append(markable)

	# Order markables in each group to ensure backwards chain
	# except in the case of cataphors, which point forward
	for group in marks_by_group:
		last_mark = None
		for mark in marks_by_group[group]:
			if mark.coref_type != "cata":
				if last_mark is not None:
					mark.antecedent = last_mark
				last_mark = mark

	# Check for markables to remove in postprocessing
	if len(lex.filters["remove_head_func"].pattern) > 0:
		for mark in markables:
			if lex.filters["remove_head_func"].match(mark.head.func) is not None and (mark.form != "proper" or
						mark.entity == "abstract" or
						(mark.text in ["U.S.","US"] and mark.func == "nn") or (mark.text in lex.first_names and mark.entity != lex.filters["time_def_entity"])): # TODO: de-hardwire Proper restriction matching OntoNotes guidelines; US is interpreted as "American" (amod); forbid abstract nn modifier even if proper
				splice_out(mark, marks_by_group[mark.group])
	if len(lex.filters["remove_child_func"].pattern) > 0:
		for mark in markables:
			for child_func in mark.head.child_funcs:
				if lex.filters["remove_child_func"].match(child_func) is not None and mark.head.func != "cata":
					splice_out(mark, marks_by_group[mark.group])
	# Handle nested compound persons
	if len(lex.rm_nested_entities) > 0:
		for mark in markables:
			if remove_nested(mark,markbyhead,lex):
				splice_out(mark, marks_by_group[mark.group])

	# Remove i in i rule (no overlapping markable coreference in group)
	# TODO: make this more efficient (iterates all pairwise comparisons)
	for group in marks_by_group:
		for mark1 in marks_by_group[group]:
			for mark2 in marks_by_group[group]:
				if not mark1 == mark2:
					if markables_overlap(mark1, mark2, None):
						if (mark1.end - mark1.start) > (mark2.end - mark2.start):
							splice_out(mark2, marks_by_group[group])
						else:
							splice_out(mark1, marks_by_group[group])

	# Remove cataphora if desired
	if lex.filters["remove_cataphora"]:
		for mark in markables:
			if mark.coref_type == "cata":
				mark.id = "0"
				if mark.antecedent != "none":
					mark.antecedent.id = "0"

	# Remove coordination envelopes if desired
	if lex.filters["remove_coordinate_envelopes"]:
		for group in marks_by_group:
			coordination_text = ""
			wipe_coord = False
			for mark in marks_by_group[group]:
				if mark.coordinate:
					coordination_text = mark.core_text
					wipe_coord = True
			if coordination_text != "":
				for mark in marks_by_group[group]:
					if mark.core_text != coordination_text:  # This group has multiple text realizations
						wipe_coord = False
			if wipe_coord:
				for mark in marks_by_group[group]:
					mark.id = "0"

	# Inactivate singletons if desired by setting their id to 0
	if lex.filters["remove_singletons"]:
		for group in marks_by_group:
			wipe_group = True
			if len(marks_by_group[group]) < 2:
				for singleton in marks_by_group[group]:
					singleton.id = "0"
			else:
				for singleton_candidate in marks_by_group[group]:
					if singleton_candidate.antecedent != 'none':
						wipe_group = False
						break
				if wipe_group:
					for singleton in marks_by_group[group]:
						singleton.id = "0"


	# Add apposition envelopes if desired
	if lex.filters["add_appos_envelopes"]:
		for group in marks_by_group:
			for i in reversed(range(1,len(marks_by_group[group]))):
				# Print marks_by_group[group]
				mark = marks_by_group[group][i]
				prev = mark.antecedent
				if prev != "none":
					if prev.coref_type == "appos" and prev.antecedent != "none":
						# Two markables in the envelop: prev and prevprev
						prevprev = prev.antecedent
						envlop = create_envelope(prevprev, prev, conll_tokens)

						# Extend markable to trailing closing punctuation if it contains opening punctuation
						if envlop.end < len(conll_tokens) - 1:
							next_id = envlop.end + 1
							lex.open_close_punct[","] = "," # Add trailing comma option to envelope to match OntoNotes behavior
							if markable_extend_punctuation(envlop.text, conll_tokens[next_id], lex.open_close_punct, "trailing"):
								envlop.text += conll_tokens[next_id].text + " "
								envlop.end += 1
							# Idiosyncratic "years old" behavior
							elif conll_tokens[envlop.end].text == "years" and conll_tokens[next_id].text == "old":
								envlop.text += conll_tokens[next_id].text + " "
								envlop.end += 1

						markables.append(envlop)
						markstart[envlop.start].append(envlop)
						markend[envlop.end].append(envlop)

						# Markables_by_head
						head_id=str(prevprev.head.id) + "_" + str(prev.head.id)
						markbyhead[head_id] = envlop

						# Set some fields for the envlop markable
						envlop.non_antecdent_groups = prev.antecedent
						# New group number for the two markables inside the envelope
						ab_group = 1000 + int(prevprev.group) + int(prev.group)
						prevprev.group = ab_group
						prev.group = ab_group
						mark.antecedent = envlop
						prevprev.antecedent = "none"

	kill_zero_marks(markables, markstart, markend)


def kill_zero_marks(markables, markstart_dict, markend_dict):
	"""
	Removes markables whose id has been set to 0 in postprocessing
	
	:param markables: All Markable objects
	:param markstart_dict: Dictionary of token span start ids to lists of markables starting at that id
	:param markend_dict: Dictionary of token span end ids to lists of markables ending at that id
	:return: void
	"""
	marks_to_kill = []
	for mark in markables:
		if mark.id == "0":  # Markable has been marked for deletion
			markstart_dict[mark.start].remove(mark)
			if len(markstart_dict[mark.start]) < 1:
				del markstart_dict[mark.start]
			markend_dict[mark.end].remove(mark)
			if len(markend_dict[mark.start]) < 1:
				del markend_dict[mark.start]
			marks_to_kill.append(mark)

	for mark in marks_to_kill:
		markables.remove(mark)


def splice_out(mark, group):
	min_id = 0
	mark_id = int(mark.id.replace("referent_", ""))
	for member in group:
		if member.antecedent == mark:
			member.antecedent = mark.antecedent
		member_id = int(member.id.replace("referent_", ""))
		if (min_id == 0 or min_id > member_id) and member.id != mark.id:
			min_id = member_id
	mark.antecedent = "none"
	if str(mark_id) != mark.group:
		mark.group = str(mark_id)
	else:
		for member in group:
			if member != mark:
				member.group = str(min_id)
	mark.id = "0"


def create_envelope(first,second, conll_tokens):
	mark_id="env"
	form = "proper" if (first.form == "proper" or second.form == "proper") else "common"
	head=first.head
	definiteness=first.definiteness
	start=first.start
	end=second.end
	intermediate = ""
	if first.end+1 < second.start: # Some intervening tokens should be included in the envelope text
		for tok in conll_tokens[first.end+1:second.start]:
			intermediate += tok.text + " "
	text=first.text.strip() + " " + intermediate + second.text.strip()
	entity=second.entity
	entity_certainty=second.entity_certainty
	subclass=first.subclass
	infstat=first.infstat
	agree=first.agree
	sentence=first.sentence
	antecedent=first.antecedent
	coref_type=first.coref_type
	group=first.group
	alt_entities=first.alt_entities
	alt_subclasses=first.alt_subclasses
	alt_agree=first.alt_agree
	cardinality=0
	if first.cardinality!=0:
		if first.cardinality == second.cardinality:
			cardinality = first.cardinality

	envelope = Markable(mark_id, head, form, definiteness, start, end, text, text, entity, entity_certainty, subclass, infstat, agree, sentence, antecedent, coref_type, group, alt_entities, alt_subclasses, alt_agree, cardinality)

	return envelope


def remove_nested(mark,markbyhead,lex):
	for nested_entity, func, container_entity in lex.rm_nested_entities:
		if mark.func == func and mark.entity == nested_entity:
			if mark.head.head in markbyhead:
				container = markbyhead[mark.head.head]
				if container.entity == container_entity:
					return True
	return False
