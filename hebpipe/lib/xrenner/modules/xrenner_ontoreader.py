from collections import defaultdict
import re, sys
import xml.dom.minidom


def printToFile(content, filename):
	firstname = filename.split(".")[0]
	newname = firstname + "_cl.coref"
	f = open(newname, 'w')
	f.write(content)
	f.close()


def removeEmptyCat(filename):
	f = open(filename, 'r')
	raw = f.read()
	insBefore = addNewlineBefore(raw)
	insAfter = addNewlineAfter(insBefore)

	tokens = re.split(' |\n', insAfter)
	new = ""
	for tok in tokens:
		star = re.search(r"(^|[^\\])\*", tok)
		if (not star and tok != "0") or tok.count("*") > 2 or (
				tok.count("*") < 3 and re.search(r'\*[a-z][a-z][a-z][a-z]+', tok) is not None):
			new = new + " " + tok

	return new


def addNewlineBefore(text):
	tokens = text.split("<")
	new = ""
	for tok in tokens[1:]:
		new = new + "\n" + "<" + tok
	return new


def addNewlineAfter(text):
	# same approach to add a \n after >
	tokens = text.split(">")
	new = ""
	for tok in tokens:
		new = new + tok + ">" + "\n"
	return new


def mainPreprocess(filen):
	"""preprocess OntoNotes gold files by cleaning up the empty categories and more"""
	# filen=sys.argv[1]
	print "Processing " + filen
	noemp = removeEmptyCat(filen)
	noempi = noemp[:-2]
	a = xml.dom.minidom.parseString(noempi)
	# pretty_xml_as_string = a.toprettyxml()
	return a


class OntoMark:
	def __init__(self, start, end, text, group):
		self.start = start
		self.end = end
		self.text = text.strip()
		self.group = group
		self.span = str(start) + "-" + str(end)

	def __repr__(self):
		return "(" + str(self.start) + "," + str(self.end) + ") group:" + str(self.group) + " text:" + self.text


class Oracle():

	def __init__(self):
		self.correct_pairs = []
		self.incorrect_triples = [] # Triples of anaphor - wrong_antecedent - correct antecedent

	def process_onto_mark(self, node, start):
		"""
		:param node: an Element node representing a markable
		:param start: the token number of the first tok
		:return: a list whose first item if number of tokens in this element, and following items are OntoMark objects
				representing this markable and any embedded within it (found recursively)
		"""
		project_appos_wrapper = False
		group = node.getAttribute("ID")
		if len(node.childNodes) == 1: # node is just one markable, child should be a Text node
			text = node.firstChild.data
			split_text = text.split()
			end = start + len(split_text) - 1

			return [len(split_text), OntoMark(start, end, text, group)]
		else:
			text = ""
			token_counter = 0
			child_marks = []
			for child in node.childNodes:
				if child.nodeType == 3: # Text
					text += child.data
					token_counter = len(text.split())
				elif child.nodeType == 1:
					if project_appos_wrapper:
						if child.getAttribute("TYPE") == "APPOS": # fix appos wrapper issue
							child.setAttribute("ID", node.getAttribute("ID"))
					process_child = self.process_onto_mark(child, start + token_counter)
					text += process_child[1].text
					child_marks += process_child[1:]
					token_counter = len(text.split())
			return [token_counter, OntoMark(start, start + token_counter - 1, text, group)] + child_marks



	def get_gold_data(self, docname, coref_dir="C:\\Uni\\CL\\xrenner\\resources\\anto_emma\\"):

		xml_filename = coref_dir + docname + ".coref"

		# preprocess and open xml file
		xml_file = mainPreprocess(xml_filename)
		xml_list = xml_file.childNodes[0].childNodes[1].childNodes

		self.group_dict = defaultdict(list)

		self.mark_dict = {}

		# get a list of  onto markables to cross-check with later
		xml_tok_counter = 1
		xml_marks = []
		for node in xml_list:
			if xml_tok_counter > 45:
				pass
			if node.nodeType == 3: # text
				s = node.data.split()
				xml_tok_counter += len(s)
			elif node.nodeType == 1: # COREF element
				processed_mark = self.process_onto_mark(node, xml_tok_counter)
				xml_tok_counter += processed_mark[0]
				new_marks = processed_mark[1:]
				for new_mark in new_marks:
					xml_marks.append(new_mark)
					self.group_dict[new_mark.group].append(new_mark)
					self.mark_dict[str(new_mark.start) + "-" + str(new_mark.end)] = new_mark


	def eval_gold(self, xrenner):
		self.get_gold_data(xrenner.docname)
		out_correct = open("c:\\uni\\cl\\xrenner\\resources\\oracle_correct.tab",'w')
		out_incorrect = open("c:\\uni\\cl\\xrenner\\resources\\oracle_incorrect.tab",'w')

		for mark in xrenner.markables:
			mark_span = str(mark.start) + "-" + str(mark.end)
			if mark.antecedent != "none":
				antecedent = mark.antecedent
				antecedent_span = str(antecedent.start) + "-" + str(antecedent.end)
			else:
				antecedent_span = ""

			if mark_span in self.mark_dict:
				onto_mark = self.mark_dict[mark_span]
				mark_gold_group = onto_mark.group
				mark_group_position = self.group_dict[mark_gold_group].index(onto_mark)
				if mark_group_position == 0:
					onto_antecedent_span = ""
					onto_antecedent_text = "FIRST"
				else:
					onto_antecedent_span = self.group_dict[mark_gold_group][mark_group_position-1].span
					onto_antecedent_text = self.mark_dict[onto_antecedent_span].text

				if antecedent_span == onto_antecedent_span == "":
					# Correct antecedent
					self.correct_pairs.append(mark.text + "\tFIRST")
				elif antecedent_span == onto_antecedent_span:
					# Correct FIRST
					self.correct_pairs.append(mark.text + "\t" + mark.antecedent.text)
				elif antecedent_span not in self.mark_dict or onto_antecedent_span != antecedent_span:
					# False antecedent
					if mark.antecedent == "none":
						self.incorrect_triples.append(mark.text + "\t" + mark.head.text + "\tFIRST\tFIRST\t" + onto_antecedent_text)
					else:
						self.incorrect_triples.append(mark.text + "\t" + mark.head.text + "\t" + mark.antecedent.text + "\t" + mark.antecedent.head.text + "\t" + onto_antecedent_text)
			else:
				# False positive
				if mark.antecedent == "none":
					#self.incorrect_triples.append(mark.text + "\tFIRST\tSINGLETON")
					pass
				else:
					if mark.text == "I":
						pass
					self.incorrect_triples.append(mark.text + "\t" + mark.head.text + "\t" + mark.antecedent.text + "\t" + mark.antecedent.head.text + "\tSINGLETON")

		for item in self.correct_pairs:
			out_correct.write(item + "\n")

		for item in self.incorrect_triples:
			out_incorrect.write(item + "\n")
