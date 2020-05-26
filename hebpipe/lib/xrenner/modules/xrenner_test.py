"""
Module to generate and run unit tests

Author: Amir Zeldes
"""

from collections import defaultdict
import unittest
import re, os, sys
from .xrenner_xrenner import Xrenner
from .xrenner_coref import find_antecedent

if sys.version_info[0] < 3:
	python_version = 2
else:
	python_version = 3


def generate_test(conll_tokens, markables, parse, model="eng", name="test"):
	tok_count = len(conll_tokens)
	mark_count = 0
	ids = []
	marks_by_id = {}

	# Collect markable groups, assign IDs by extension and count markables
	marks_by_group = defaultdict(list)
	for mark in markables:
		mark_count += 1

		# Assign predictable ID of the form start_end
		id_ = str(mark.start) + "_" + str(mark.end)
		if id_ in ids:
			raise("xrenner generated two markables with same extension: tok" + str(mark.start) + ":tok" + str(mark.end))
		else:
			ids.append(id_)
			mark.id = id_
			marks_by_id[id_] = mark

		marks_by_group[int(mark.group)].append(mark)

	group_count = len(marks_by_group)

	# Serialize group details
	chains = []
	for group in sorted(marks_by_group):
		chain = sorted(marks_by_group[group],key=lambda x: int(x.id[:x.id.find("_")]))
		gid = "g" + chain[0].id
		chain_string = "  "
		for mark in chain:
			chain_string += mark.id + " < "
		chains.append(chain_string[:-3])

	chains.sort(key=lambda x: int(x[2:x.find("_")]))

	snippets = []
	for chain in chains:
		first = chain[2:chain.find("<")-1] if "<" in chain else chain.strip()
		snippet = marks_by_id[first].text[:20] + "..." if len(marks_by_id[first].text) > 20 else marks_by_id[first].text
		snippets.append(snippet)

	zipped = zip(snippets,chains)

	output = ""
	output += "name:" + name + "\n"
	output += "model:" + model + "\n"
	output += "toks:" + str(tok_count) + " # " + " ".join(tok.text for tok in conll_tokens[1:4]) + "..." + "\n"
	output += "marks:" + str(mark_count) + "\n"
	output += "groups:" + str(group_count) + "\n"
	output += "chains:" + "\n"
	for chain in zipped:
		output += "  # " + str(chain[0]) + "\n"
		output += chain[1] + "\n"
	output += "input_data:" + "\n"
	output += "\n".join(parse)
	output += "\n" + "-"*5 + "\n"
	return output


def setUpModule():
	# Read test/tests.dat
	print("\nxrenner unit tests\n" + "=" * 20 + "\nReading test cases from test/tests.dat")
	file = os.path.dirname(os.path.realpath(__file__)) + os.sep + ".." + os.sep + "test" + os.sep + "tests.dat"
	test_data = ""
	with open(file, 'rb') as f:
		test_data = f.read()

	# Populate cases with Case objects
	global cases
	cases = {}
	if python_version < 3:
		case_list = test_data.split("-----")
	else:
		case_list = test_data.decode().split("-----")

	for case in case_list:
		case = case.strip()
		if len(case) > 0:
			case_to_add = Case(case)
			cases[case_to_add.name] = case_to_add

	# Initialize an Xrenner object with the language model and assign to module level xrenner variable for all suites
	print("Initializing xrenner model 'eng'\n")
	global xrenner
	xrenner = Xrenner("eng",override="TEST")
	xrenner.set_doc_name("test_test")


class Test1Model(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		print("\nTesting model integrity\n" + "-"*30)
		global xrenner
		cls.xrenner = xrenner
		global cases
		cls.cases = cases

	@classmethod
	def tearDownClass(cls):
		global xrenner
		cls.xrenner = xrenner

	def test_model_files(self):
		print("Checking model files:  ")
		# Check that all model components were read and are filled as expected
		self.assertTrue(len(self.xrenner.lex.coref_rules),"check that coref_rules is full")
		self.assertTrue(len(self.xrenner.lex.entities),"check that entities is full")
		self.assertTrue(len(self.xrenner.lex.entity_heads),"check that entity_heads is full")
		self.assertTrue(len(self.xrenner.lex.pronouns),"check that pronouns is full")
		self.assertTrue(len(self.xrenner.lex.filters),"check that filters is full")

		# Optional components, should be included in default model
		self.assertTrue(len(self.xrenner.lex.names),"check that names is full")
		self.assertTrue(len(self.xrenner.lex.stop_list),"check that stop_list is full")
		self.assertTrue(len(self.xrenner.lex.open_close_punct),"check that open_close_punct is full")
		self.assertTrue(len(self.xrenner.lex.open_close_punct_rev),"check that open_close_punct_rev is full")
		self.assertTrue(len(self.xrenner.lex.entity_mods),"check that entity_mods is full")
		self.assertTrue(len(self.xrenner.lex.entity_deps),"check that entity_deps is full")
		self.assertTrue(len(self.xrenner.lex.hasa),"check that hasa is full")
		self.assertTrue(len(self.xrenner.lex.coref),"check that coref is full")
		self.assertTrue(len(self.xrenner.lex.numbers),"check that numbers is full")
		self.assertTrue(len(self.xrenner.lex.affix_tokens),"check that affix_tokens is full")
		self.assertTrue(len(self.xrenner.lex.antonyms),"check that antonyms is full")
		self.assertTrue(len(self.xrenner.lex.isa),"check that isa is full")


class Test2MarkableMethods(unittest.TestCase):

	@classmethod
	def setUpClass(cls):
		print("\n\nRunning markable method tests\n" + "-"*30)
		global xrenner
		cls.xrenner = xrenner
		global cases
		cls.cases = cases
		cls.xrenner.lex.filters["remove_singletons"] = False

	@classmethod
	def tearDownClass(cls):
		global xrenner
		cls.xrenner = xrenner
		cls.xrenner.lex.filters["remove_singletons"] = True

	def test_name(self):
		# Jerry B. Clinton
		print("\nRun markable name test:  ")
		target = self.cases["mark_name_test"]
		self.xrenner.analyze(target.parse.split("\n"), "unittest")
		markables = self.xrenner.markables
		# Check that there are no nested markables in Jerry B. Clinton
		self.assertEqual(len(markables),1)
		# Check that the name is classified as a person
		self.assertEqual(markables[0].entity, "person")

	def test_atomic_mod(self):
		# Israel Machines Corp.
		# Note that Corp. is an organization-marking atomic flagged modifier of the head
		print("\nRun atomic modifier test:  ")
		target = self.cases["mark_atomic_mod_test"]
		self.xrenner.analyze(target.parse.split("\n"), "unittest")
		markables = self.xrenner.markables
		# Check that there are no nested markables in Israel Machines Corp.
		self.assertEqual(len(markables), 1)
		# Check that the name is classified as a person
		self.assertEqual(markables[0].entity, "organization")


class Test3CorefMethods(unittest.TestCase):

	@classmethod
	def setUpClass(cls):

		print("\n\nRunning coref method tests\n" + "-"*30)
		global xrenner
		cls.xrenner = xrenner
		global cases
		cls.cases = cases

	def test_cardinality(self):
		# I saw two birds . The three birds flew .
		print("\nRun cardinality test:  ")
		target = self.cases["cardinality_test"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(0,result.mark_count,"cardinality test (two birds != the three birds)")

	def test_appos_envelope(self):
		# Meet [[Mark Smith] , [the Governor]]. [He] is the best.
		print("\nRun apposition envelope test:  ")
		target = self.cases["appos_envelope"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(target.chains,result.chains,"appos envelope test")

	def test_isa(self):
		# I read [the Wall Street Journal]. [That newspaper] is great.
		print("\nRun isa test:  ")
		target = self.cases["isa_test"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(target.chains,result.chains,"isa test (Wall Street Journal <- newspaper)")

	def test_hasa(self):
		# The [[CEO] and the taxi driver] ate . [[His] employees] joined them
		print("\nRun hasa test:  ")
		target = self.cases["hasa_test"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(target.chains,result.chains,"hasa test (CEO, taxi driver <- his employees)")

	def test_dynamic_hasa(self):
		# Beth was worried about [[Sinead 's] well-being] , and also about Jane . [[Her] well-being] was always a concern .
		print("\nRun dynamic hasa test:  ")
		target = self.cases["dynamic_hasa_test"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(target.chains,result.chains,"dynamic hasa test (Sinead 's <- her)")

	def test_entity_dep(self):
		# I have a book , [a dog] and a car. [It] barked.
		print("\nRun entity dep test:  ")
		target = self.cases["entity_dep_test"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(target.chains,result.chains,"entity dep test (a book, a dog <- It barked)")

	def test_affix_morphology(self):
		# [A blorker] had a mummelhound in a blargmobile. I saw [the person] .
		print("\nRun affix morphology test:  ")
		target = self.cases["morph_test"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(target.chains,result.chains,"affix morph test (A blorker <- the person)")

	def test_verbal_event_stem(self):
		# John [visited] Spain . [The visit] went well .
		print("\nRun verbal event coreference test:  ")
		target = self.cases["verb_test"]
		result = Case(self.xrenner.analyze(target.parse.split("\n"),"unittest"))
		self.assertEqual(target.chains,result.chains,"verbal event stemming (visited <- the visit	)")


class Case:

	def __init__(self, case_string):
		params, parse = case_string.split("input_data:")
		self.parse = parse.strip()
		params = params.replace("\r","")

		self.chains = []
		chain_mode = False

		for line in params.split("\n"):
			line = re.sub(r'#.*','',line).strip()
			if len(line) > 0:
				if chain_mode:
					self.chains.append(line)
				if ":" in line and not chain_mode and not "options" in line:
					key, val = line.split(":")
					if key == "name":
						self.name = val
					elif key == "toks":
						self.tok_count = int(val)
					elif key == "marks":
						self.mark_count = int(val)
					elif key == "groups":
						self.group_count = int(val)
					elif key == "model":
						self.model = val
					elif key == "chains":
						chain_mode = True


def suite():
	# Create test suite
	test_suite = unittest.TestSuite()

	# Add a test case
	test_suite.addTest(unittest.makeSuite(Test1Model))
	test_suite.addTest(unittest.makeSuite(Test2MarkableMethods))
	test_suite.addTest(unittest.makeSuite(Test3CorefMethods))

	return test_suite


def can_be_coreferent(mark1, mark2, lex):
	"""
	Utility function to check whether an xrenner model is capable of finding two markables coreferent

	:param mark1: The :class:`.Markable` object to match to mark2
	:param mark2: The :class:`.Markable` object to match to mark1
	:param lex: the :class:`.LexData` object with gazetteer information and model settings
	:return: bool
	"""

	lex.incompatible_isa_pairs = set([])
	lex.incompatible_mod_pairs = set([])
	prev_markables = [mark1, mark2]
	mark2.sentence.sent_num = 2
	antecedent, propagation = find_antecedent(mark2, prev_markables,lex)
	return antecedent is not None


if __name__ == '__main__':
	xrenner = None
	cases = {}

	test_runner = unittest.TextTestRunner()
	test_runner.run(suite())



