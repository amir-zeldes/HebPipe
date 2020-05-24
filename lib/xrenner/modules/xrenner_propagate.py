"""
modules/xrenner_propagate.py

Feature propagation module. Propagates entity and agreement features for coreferring markables.

Author: Amir Zeldes
"""

def propagate_entity(markable, candidate, direction="propagate"):
	"""
	Propagate class and agreement features between coreferent markables
	
	:param markable: a Markable object
	:param candidate: a coreferent antecedent Markable object
	:param direction: propagation direction; by default, data can be propagated in either direction from the more certain markable to the less certain one, but direction can be forced, e.g. 'propagate_forward'
	:return: void
	"""
	# Check for rule explicit instructions
	if direction == "propagate_forward":
		markable.entity = candidate.entity
		markable.subclass = candidate.subclass
		markable.entity_certainty = "propagated"
		propagate_agree(candidate, markable)
	elif direction == "propagate_back":
		candidate.entity = markable.entity
		candidate.subclass = markable.subclass
		candidate.entity_certainty = "propagated"
		propagate_agree(markable, candidate)
	else:
		# Prefer nominal propagates to pronoun
		if markable.form == "pronoun" and candidate.entity_certainty != "uncertain":
			markable.entity = candidate.entity
			markable.subclass = candidate.subclass
			propagate_agree(candidate, markable)
			markable.entity_certainty = "propagated"
		elif candidate.form == "pronoun" and markable.entity_certainty != "uncertain":
			candidate.entity = markable.entity
			candidate.subclass = markable.subclass
			candidate.entity_certainty = "propagated"
			propagate_agree(markable, candidate)
		else:
			# Prefer certain propagates to uncertain
			if candidate.entity_certainty == "uncertain":
				candidate.entity = markable.entity
				candidate.subclass = markable.subclass
				candidate.entity_certainty = "propagated"
				propagate_agree(markable, candidate)
			elif markable.entity_certainty == "uncertain":
				markable.entity = candidate.entity
				markable.subclass = candidate.subclass
				markable.entity_certainty = "propagated"
				propagate_agree(candidate, markable)
			else:
				# Prefer to propagate to satisfy alt_entity
				if markable.entity != candidate.entity and markable.entity in candidate.alt_entities:
					candidate.entity = markable.entity
					candidate.subclass = markable.subclass
					candidate.entity_certainty = "certain"
					propagate_agree(markable, candidate)
				elif markable.entity != candidate.entity and candidate.entity in markable.alt_entities:
					markable.entity = candidate.entity
					markable.subclass = candidate.subclass
					markable.entity_certainty = "certain"
					propagate_agree(candidate, markable)
				else:
					# Prefer to propagate backwards
					candidate.entity = markable.entity
					candidate.subclass = markable.subclass
					candidate.entity_certainty = "propagated"
					propagate_agree(markable, candidate)


def propagate_agree(markable, candidate):
	"""
	Progpagate agreement between to markables if one has unknown agreement
	
	:param markable: Markable object
	:param candidate: Coreferent antecdedent Markable object
	:return: void
	"""
	if (candidate.agree == '' or candidate.agree is None) and not (markable.agree == '' or markable.agree is None):
		candidate.agree = markable.agree
	else:
		markable.agree = candidate.agree
