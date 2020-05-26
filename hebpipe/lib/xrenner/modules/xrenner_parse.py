# Requirements:
# spacy + English model
# xrenner (pip install xrenner)
# depedit (pip install depedit)

from spacy.en import English

# Part 1: Use spacy to get a dependency parse of the text
text = u"The CEO and the taxi driver sat down. His employees joined them."
parser = English(entity=False,load_vectors=False,vectors_package=False)
parsed = parser(text)


# Part 2: xrenner expects Stanford typed dependencies
# We can get these by running the Stanford Parser with 'basic' dependencies in 'conllx' format
# Or we can use spacy, slightly rewire the results and labels, and construct conll format, as follows:
parsed_string=""
toks = 0
prev_s_toks = 0
for sent in parsed.sents:
	toks = 0
	for index, token in enumerate(sent):
		if token.head.i+1 - prev_s_toks == token.i+1 - prev_s_toks: # Spacy represents root as self-dominance, revert to 0
			head_id = 0
		else:
			head_id = token.head.i + 1 - prev_s_toks

		line = [str(token.i+1 - prev_s_toks),token.orth_,token.lemma_,token.tag_,token.pos_,"_", str(head_id), token.dep_, "_","_"]
		parsed_string += "\t".join(line) + "\n"
		toks += 1
	prev_s_toks += toks
	parsed_string += "\n"

# The parse string now looks like this:
print "Original spacy parse, in conll format:"
print parsed_string

# Part 3: editing the labels to be Stanford-like
# This part uses depedit: a module to do 'find and replace' in dependency trees
# To learn more about how it works, see the user guide at: http://corpling.uis.georgetown.edu/depedit

from depedit import DepEdit

config =  "func=/ROOT/\tnone\t#1:func=root\n"
config += "func=/relcl/\tnone\t#1:func=rcmod\n"
config += "func=/nummod/\tnone\t#1:func=num\n"
config += "lemma=/be/&func=/(.*)/;func=/nsubj/;func=/attr/;text=/.*/\t#1>#2;#1>#3;#4>#1\t#4>#3;#3>#1;#3>#2;#1:func=cop;#3:func=$1\n"
config += "lemma=/be/&func=/root/;func=/nsubj/;func=/attr/\t#1>#2;#1>#3\t#3>#1;#3>#2;#1:func=cop;#3:func=root\n"

deped = DepEdit(config.split("\n"))

edited = deped.run_depedit(parsed_string.split("\n"))

# The parse string now looks like this:
print "Edited parse, after depedit:"
print edited


# Part 4: actual coref stuff
from xrenner.modules.xrenner_xrenner import Xrenner

# Load a model to initialize the Xrenner object with language specific lexical information
xrenner = Xrenner(model="C:\\eng.xrm",override="GUM")  # If you omit the gum override, you will get the OntoNotes schema, which ignores singleton mentions among other things

# Output the analysis in conll coref format - you can also try "html" or "sgml", among other options
print xrenner.analyze(edited, "conll")




####### Object Model Examples ########
# Heres how to access some properties of the object model once the analysis has been run:

all_markables = xrenner.markables
fourth_markable = all_markables[3]  # This is "His"
print fourth_markable  # Get a textual summary of this markable

# Get the lemma of the antecedent for fourth markable if it's given
if fourth_markable.infstat == "giv":
	print fourth_markable.antecedent.lemma  # Note that the antecedent is a markable object, and has all the same properties
	# Compare grammatical functions for anaphor and antecedent if they are not in the same sentence
	if fourth_markable.sentence.sent_num == fourth_markable.antecedent.sentence.sent_num:
		print "Antecedent is in same sentence"
	else:
		if fourth_markable.func == fourth_markable.antecedent.func:
			print "The anaphor retains the same function: " + fourth_markable.func
		else:
			print "Anaphor function is: " + fourth_markable.func + " but antecedent is: " + fourth_markable.antecedent.func
else:
	print "The markable's information status is " + fourth_markable.infstat

# See the slides for some other markable properties you can play with - there are many more



## PS - xrenner can also be fed a text file containing a parse from the file system like this:
#print xrenner.analyze("C:\\parses\\example_parse.conll10","conll")