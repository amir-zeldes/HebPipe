# HebPipe Hebrew NLP Pipeline

A simple NLP pipeline for Hebrew text in UTF-8 encoding, using standard components. Basic features:

  * Performs end to end processing, optionally skipping steps as needed:
    * whitespace tokenization
    * morphological segmentation
    * POS tagging
    * morphological tagging
    * dependency parsing
    * named and non-named entity type recognition (**experimental**)
    * coreference resolution (**experimental**)
  * Does not alter the input string (text reconstructible from, and alignable to output)
  * Compatible with Python 3.5+, Linux, Windows and OSX

Note that entity recognition and coreference are still in beta and offer rudimentary accuracy.

To cite this tool in academic papers please refer to this paper:

Zeldes, Amir, Nick Howell, Noam Ordan and Yifat Ben Moshe (2022) [A Second Wave of UD Hebrew Treebanking and Cross-Domain Parsing](https://arxiv.org/abs/2210.07873). In: *Proceedings of EMNLP 2022*. Abu Dhabi, UAE.


```
@InProceedings{ZeldesHowellOrdanBenMoshe2022,
  author    = {Amir Zeldes and Nick Howell and Noam Ordan and Yifat Ben Moshe},
  booktitle = {Proceedings of {EMNLP} 2022},
  title     = {A SecondWave of UD Hebrew Treebanking and Cross-Domain Parsing},
  pages     = {4331--4344},
  year      = {2022},
  address   = {Abu Dhabi, UAE},
}
```

## Performance

Current scores on UD_Hebrew-HTB (IAHLT version tokenization) using the official conll scorer, end to end from plain text, trained jointly on UD Hebrew:

```
Metric     | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |     99.93 |     99.97 |     99.95 |
Sentences  |     98.39 |     99.39 |     98.89 |
Words      |     99.14 |     99.09 |     99.11 |
UPOS       |     96.17 |     96.12 |     96.15 |     97.01
XPOS       |     96.17 |     96.12 |     96.15 |     97.01
UFeats     |     90.25 |     90.21 |     90.23 |     91.04
AllTags    |     89.61 |     89.57 |     89.59 |     90.39
Lemmas     |     95.26 |     95.21 |     95.23 |     96.09
UAS        |     90.45 |     90.41 |     90.43 |     91.24
LAS        |     87.64 |     87.60 |     87.62 |     88.41
CLAS       |     82.82 |     82.33 |     82.57 |     83.39
MLAS       |     69.68 |     69.27 |     69.47 |     70.16
BLEX       |     78.01 |     77.55 |     77.78 |     78.55
```

Current scores on UD_Hebrew-IAHLTwiki using the official conll scorer, end to end from plain text, trained jointly on UD Hebrew:

```
Metric     | Precision |    Recall |  F1 Score | AligndAcc
-----------+-----------+-----------+-----------+-----------
Tokens     |     99.71 |     99.89 |     99.80 |
Sentences  |     99.49 |     99.75 |     99.62 |
Words      |     99.48 |     99.19 |     99.33 |
UPOS       |     96.57 |     96.29 |     96.43 |     97.08
XPOS       |     96.57 |     96.29 |     96.43 |     97.08
UFeats     |     90.90 |     90.63 |     90.77 |     91.38
AllTags    |     90.21 |     89.95 |     90.08 |     90.69
Lemmas     |     97.37 |     97.09 |     97.23 |     97.89
UAS        |     92.44 |     92.17 |     92.31 |     92.93
LAS        |     90.08 |     89.82 |     89.95 |     90.56
CLAS       |     86.48 |     85.82 |     86.15 |     86.81
MLAS       |     73.04 |     72.49 |     72.76 |     73.32
BLEX       |     83.61 |     82.98 |     83.29 |     83.93
```

## Installation

Either install from PyPI using pip:

`pip install hebpipe`

And run as a module:

`python -m hebpipe example_in.txt`

Or install manually: 

  * Clone this repository into the directory that the script should run in (git clone https://github.com/amir-zeldes/HebPipe)
  * In that directory, install the dependencies under **Requirements**, e.g. by running `python setup.py install` or `pip install -r requirements.txt`
  
Models can be downloaded automatically by the script on its first run.
  
## Requirements

### Python libraries

Required libraries:

```
requests
transformers==4.35.2
torch==2.1.0
xgboost==2.0.3
gensim==4.3.2
rftokenizer>=2.2.0
numpy
scipy
depedit>=3.3.1
pandas==2.1.2
joblib==1.3.2
xmltodict==0.13.0
diaparser==1.1.2
flair==0.13.0
stanza==1.7.0
conllu==4.5.3
protobuf==4.23.4
```

You should be able to install these manually via pip if necessary (i.e. `pip install rftokenizer==2.2.0` etc.).

Note that some older versions of Python + Windows do not install numpy correctly from pip, in which case you can download compiled binaries for your version of Python + Windows here: https://www.lfd.uci.edu/~gohlke/pythonlibs/


### Model files

Model files are too large to include in the standard GitHub repository. The software will offer to download them automatically. The latest models can also be downloaded manually at https://gucorpling.org/amir/download/heb_models_v4/. 

## Command line usage

```
usage: python heb_pipe.py [OPTIONS] files

positional arguments:
  files                 File name or pattern of files to process (e.g. *.txt)

options:
  -h, --help            show this help message and exit

standard module options:
  -w, --whitespace      Perform white-space based tokenization of large word forms
  -t, --tokenize        Tokenize large word forms into smaller morphological segments
  -p, --posmorph        Do POS and Morph tagging
  -l, --lemma           Do lemmatization
  -d, --dependencies    Parse with dependency parser
  -e, --entities        Add entity spans and types
  -c, --coref           Add coreference annotations
  -s SENT, --sent SENT  XML tag to split sentences, e.g. "s" for <s ..> ... </s>, or "newline" to use newlines, "auto" for automatic splitting, or "both" for both
  -o {pipes,conllu,sgml}, --out {pipes,conllu,sgml}
                        Output CoNLL format, SGML or just tokenize with pipes

less common options:
  -q, --quiet           Suppress verbose messages
  -x EXTENSION, --extension EXTENSION
                        Extension for output files (default: .conllu)
  --cpu                 Use CPU instead of GPU (slower)
  --disable_lex         Do not use lexicon during lemmatization
  --dirout DIROUT       Optional output directory (default: this dir)
  --from_pipes          Input contains subtoken segmentation with the pipe character (no automatic tokenization is performed)
  --version             Print version number and quit
```

### Example usage

Whitespace tokenize, tokenize morphemes, add pos, lemma, morph, dep parse with automatic sentence splitting,
entity recognition and coref for one text file, output in default conllu format:
> python heb_pipe.py -wtpldec example_in.txt

OR specify no processing options (automatically assumes you want all steps)
> python heb_pipe.py example_in.txt

Just tokenize a file using pipes:
> python heb_pipe.py -wt -o pipes example_in.txt

POS tag, lemmatize, add morphology and parse a pre-tokenized file, splitting sentences by existing <sent> tags:
> python heb_pipe.py -pld -s sent example_in.txt

Add full analyses to a whole directory of *.txt files, output to a specified directory:
> python heb_pipe.py -wtpldec --dirout /home/heb/out/ *.txt

Parse a tagged TT SGML file into CoNLL tabular format for treebanking, use existing tag <sent> to recognize sentence borders:
> python heb_pipe.py -d -s sent example_in.tt

## Input formats

The pipeline accepts the following kinds of input:

  * Plain text, with normal Hebrew whitespace behavior. Newlines are assumed to indicate a sentence break, but longer paragraphs will receive automatic sentence splitting too (use: -s both).
  * Gold super-tokenized: if whitespace tokenization is already done, you can leave out `-w`. The system expect one super-token per line in this case (e.g. בבית is on one line)
  * Gold tokenized: if gold morphological segmentation is already done, you can input one gold token per line.
  * Pipes: if morphological segmentation is already done, you can also input one super-token per line with sub-tokens separated by pipes - use `--from_pipes` for this option (allows running the segmenter, outputting pipes for manual correction, then continuing NLP processing from pipes)
  * XML sentence tags in input: use -s TAGNAME to indicate an XML tag providing gold sentence boundaries.
