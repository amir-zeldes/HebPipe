import re, operator

class CorefRule:
	def __init__(self,rule_string, rule_num):
		if not 3 <= rule_string.count(";") <= 5:
			raise Exception("coref rule does not contain 3-5 semicolons: " + rule_string)
		parts = rule_string.split(";")
		self.ana_spec, self.ante_spec, self.max_distance, self.propagation = parts[0:4]
		if len(parts) > 4:
			self.clf_name = parts[4]
		else:
			self.clf_name = "_default_"
		if len(parts) == 6:
			self.thresh = float(parts[5])
		else:
			self.thresh = None
		self.max_distance = int(self.max_distance)
		self.ana_constraints = []
		self.ante_constraints = []
		for item in self.ana_spec.split("&"):
			self.ana_constraints.append(ConstraintMatcher(item))
		for item in self.ante_spec.split("&"):
			self.ante_constraints.append(ConstraintMatcher(item))
		# Make sure that group failure criteria are first to be checked
		self.ante_constraints.sort(key=lambda x: x.group_failure, reverse=True)
		self.rule_num = rule_num

	def __repr__(self):
		return self.ana_spec + " -> " + self.ante_spec + " (" + str(self.max_distance) + ", " + self.propagation + ", "+ self.clf_name + ")"


class ConstraintMatcher:
	def __init__(self,constraint):
		self.group_failure = False
		self.negative = operator.truth
		self.match_type = "exact"
		self.value = ""
		self.key = ""
		self.compiled_re = None
		self.props = {"form", "text", "agree", "entity", "subclass", "cardinality","text_lower","lemma","pos","func","quoted","mood","speaker"}

		if constraint.endswith("*"):
			self.group_failure = True
			constraint = constraint[0:-1]

		if "=" in constraint:  # Key-value constrains
			key, value = constraint.split("=")
			if key[-1] == "!":
				self.negative = operator.not_
				key = key[0:-1]

			if value.startswith('"') and value.endswith('"'):
				value = value[1:-1]
				self.match_type = "exact"
			elif value.startswith("/") and value.endswith("/"):
				value = value[1:-1]
				if re.escape(value) == value:  # Does not contain special characters but is medial regex
					self.match_type = "exact"
				elif re.escape(value[1:-1]) == value and value.startswith("^") and value.endswith("$"):
					# Regex only supplies anchors, treat as exact match
					value = value[1:-1]
					self.match_type = "exact"
				elif re.escape(value[1:]) == value and value.startswith("^"):
					# Regex only supplies initial anchor, treat as startswith
					value = value[1:]
					self.match_type = "startswith"
				elif re.escape(value[:-1]) == value and value.endswith("$"):
					# Regex only supplies initial anchor, treat as startswith
					value = value[:-1]
					self.match_type = "endswith"
				else:
					self.match_type = "regex"
					self.compiled_re = re.compile(value)
			elif value.lower() == "true":
				self.match_type = "bool"
				self.value = True
			elif value.lower() == "false":
				self.match_type = "bool"
				self.value = False
			elif value.startswith("$"):
				# Antecedent-based spec, can't precompile value matcher
				self.match_type = "dollar"
			else:  # String literal without regex
				self.match_type = "exact"
			if self.match_type != "bool":
				self.value = value
			self.key = key
		elif constraint == "none" or constraint.startswith("any") or constraint.startswith("look") or constraint.startswith("take"):
			# This is a 'none' matcher or processing instruction, matches anything
			self.match_type = "none"
		elif "sameparent" in constraint:  # same or !same style constraint - port to $ for backwards compatibility
			if constraint[0] == "!":
				self.negative = operator.not_
			self.match_type = "dollar"
			self.key = "parent"
			self.value = "$1"
		elif "samespeaker" in constraint:  # same or !same style constraint - port to $ for backwards compatibility
			if constraint[0] == "!":
				self.negative = operator.not_
			self.match_type = "dollar"
			self.key = "speaker"
			self.value = "$1"
		elif constraint.startswith("last["):
			self.match_type = "exact"
			self.key = "LAST"
			self.value = constraint[constraint.find("[")+1:-1]

	def __repr__(self):
		if self.negative == operator.truth:
			op = ""
		else:
			op = "!"
		return self.key + " " + op + self.match_type + " '" + self.value + "'"

	def match(self,mark,lex,anaphor=None):
		test_val = ""
		op = self.negative

		if self.match_type == "none":
			return True
		elif self.match_type == "dollar":
			if self.key in self.props:
				self.value = str(getattr(anaphor, self.key))
				test_val = str(getattr(mark, self.key))
			elif self.key == "head":
				return op(anaphor.head.id == mark.head.head)
			elif self.key == "child":
				return op(anaphor.head.head == mark.head.id)
			elif self.key == "hasa":
				return op(anaphor.head.head_text in lex.hasa[mark.lemma])
			elif self.key == "parent":
				if mark.head.head == "0":  # Root token, by definition not same parent as another token
					retval = op(False)
				elif mark.sentence.sent_num != anaphor.sentence.sent_num:
					retval = op(False)
				else:
					retval = op(anaphor.head.head == mark.head.head)
				if retval is False and self.group_failure and anaphor is not None:
					mark.non_antecdent_groups.add(anaphor.group)
					anaphor.non_antecdent_groups.add(mark.group)
				return retval
			elif self.key == "has_child_func":
				raise Exception("coref rule 'has_child_func=$' : $ identity not implemented for has_child_func")
			elif self.key == "mod":
				mods = getattr(anaphor.head, "modifiers")
				found_mod = False
				for mod1 in mark.head.modifiers:
					for mod2 in mods:
						if mod1.lemma == mod2.lemma and lex.filters["det_func"].match(mod1.func) is None and \
						lex.filters["det_func"].match(mod2.func) is None:
							found_mod = True
				if not found_mod:
					if self.group_failure and anaphor is not None:
						mark.non_antecdent_groups.add(anaphor.group)
					return False
				else:
					return True
		else:
			if self.key in self.props:
				if self.match_type == "bool":
					test_val = getattr(mark, self.key)
				else:
					test_val = str(getattr(mark, self.key))
			elif self.key == "LAST":
				if self.value in lex.last:
					return op(lex.last[self.value].entity==mark.entity)
				else:
					return False
			elif self.key == "has_child_func":
				test_val = mark.child_func_string
				self.match_type = "substring"
				if self.value[0] != ";":
					self.value = ";"+self.value+";"
			elif self.key == "mod":
				mods = [self.value]
				found_mod = False
				for mod1 in mark.head.modifiers:
					for mod2 in mods:
						# Note that mod2 is just a string - not a ParsedToken
						if mod1.lemma == mod2 and lex.filters["det_func"].match(mod1.func) is None:
							found_mod = True
				if not found_mod:
					if self.group_failure and anaphor is not None:
						mark.non_antecdent_groups.add(anaphor.group)
					return False
				else:
					return True
			elif self.key == "head":
				raise Exception("coref rule 'head=VAL' : value match not implemented for head")
			elif self.key == "child":
				raise Exception("coref rule 'child=VAL' : value match not implemented for child")

		if self.match_type == "exact":
			retval = op(test_val == self.value)
		elif self.match_type == "substring":
			retval = op(self.value in test_val)
		elif self.match_type == "regex":
			retval = op(self.compiled_re.search(test_val) is not None)
		elif self.match_type == "startswith":
			retval = op(test_val.startswith(self.value))
		elif self.match_type == "endswith":
			retval = op(test_val.endswith(self.value))
		elif self.match_type == "dollar":
			retval = op(test_val == self.value)
		elif self.match_type == "bool":
			retval = op(test_val == self.value)

		if retval is False and self.group_failure and anaphor is not None:
			mark.non_antecdent_groups.add(anaphor.group)

		return retval

