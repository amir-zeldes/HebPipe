# HebPipe Hebrew NLP Pipeline

An simple NLP pipeline for Hebrew text in UTF-8 encoding, using standard components. Basic features:

  * Performs end to end processing, optionally skipping steps as needed:
    * whitespace tokenization
    * morphological segmentation (excl. insertion of unexpressed articles)
    * POS tagging
    * morphological tagging
    * dependency parsing
    * named and non-named entity type recognition (**experimental**)
    * coreference resolution (**experimental**)
  * Does not alter the input string (text reconstructible from, and alignable to output)
  * Compatible with Python 2.7/3.5+, Linux, Windows and OSX

Note that entity recognition and coreference are still in beta and offer rudimentary accuracy.

Online demo available at: (choose 'Hebrew' and enter plain text)

https://corpling.uis.georgetown.edu/xrenner/

To cite this work please refer to the paper about the morphological segmenter here:

Zeldes, Amir (2018) A Characterwise Windowed Approach to Hebrew Morphological Segmentation. In: *Proceedings of the 15th SIGMORPHON Workshop on Computational Research in Phonetics, Phonology, and Morphology*. Brussels, Belgium.

```
@InProceedings{,
  author    = {Amir Zeldes},
  title     = {A CharacterwiseWindowed Approach to {H}ebrew Morphological Segmentation},
  booktitle = {Proceedings of the 15th {SIGMORPHON} Workshop on Computational Research in Phonetics, Phonology, and Morphology},
  year      = {2018},
  address   = {Brussels, Belgium}
}
```

## Installation

  * Clone this repository into the directory that the script should run in (git clone https://github.com/amir-zeldes/HebPipe)
  * In that directory, install the dependencies under **Requirements**, e.g. by running `python setup.py install` or `pip install -r requirements.txt`
  
## Requirements

### Python libraries

The NLP pipeline will run on Python 2.7+ or Python 3.5+ (2.6 and lower are not supported). Required libraries:

  * requests
  * numpy
  * pandas
  * depedit
  * xrenner
  * scikit-learn==0.19.0

You should be able to install these manually via pip if necessary (i.e. `pip install scikit-learn==0.19.0` etc.).

Note that some versions of Python + Windows do not install numpy correctly from pip, in which case you can download compiled binaries for your version of Python + Windows here: https://www.lfd.uci.edu/~gohlke/pythonlibs/, then run for example:

`pip install c:\some_directory\numpy‑1.15.0+mkl‑cp27‑cp27m‑win_amd64.whl`

### External dependencies

The pipeline also requires **java** to be available (for parsing, tagging and morphological disambiguation). You will need binaries of Marmot and MaltParser 1.9.1 if you want to use POS tagging, morphology and parsing. These are not included in the distribution but **the script will offer to attempt to download them if they are missing**.

## Command line usage

```
usage: python heb_pipe.py [OPTIONS] files

positional arguments:
  files                 File name or pattern of files to process (e.g. *.txt)

optional arguments:
  -h, --help            show this help message and exit

standard module options:
  -w, --whitespace      Perform white-space based tokenization of large word
                        forms
  -t, --tokenize        Tokenize large word forms into smaller morphological
                        segments
  -p, --pos             Do POS tagging
  -l, --lemma           Do lemmatization
  -m, --morph           Do morphological tagging
  -d, --dependencies    Parse with dependency parser
  -e, --entities        Add entity spans and types
  -c, --coref           Add coreference annotations
  -s {auto,none}, --sent {auto,none}
                        XML tag to split sentences, e.g. sent for <sent ..> or
                        none for no splitting (otherwise automatic sentence
                        splitting)
  -o {pipes,conllu,sgml}, --out {pipes,conllu,sgml}
                        Output CoNLL format, SGML or just tokenize with pipes

less common options:
  -q, --quiet           Suppress verbose messages
  -x EXTENSION, --extension EXTENSION
                        Extension for output files (default: .conllu)
  --dirout DIROUT       Optional output directory (default: this dir)
  --version             Print version number and quit
```

### Example usage

Whitespace tokenize, tokenize morphemes, add pos, lemma, morph, dep parse with automatic sentence splitting, 
entity recognition and coref for one text file, output in default conllu format:
> python heb_pipe.py -wtplmdec infile.txt        

Just tokenize a file using pipes:
> python heb_pipe.py -wt -o pipes infile.txt       

Pos tag, lemmatize, add morphology and parse a pre-tokenized file, splitting sentences by existing <sent> tags:
> python heb_pipe.py -plmd -s sent infile.txt       

Add full analyses to a whole directory of *.txt files, output to a specified directory:    
> python heb_pipe.py -wtplmdec --dirout /home/heb/out/ *.txt

Parse a tagged TT SGML file into CoNLL tabular format for treebanking, use existing tag <sent> to recognize sentence borders:
> python heb_pipe.py -d -s sent infile.tt

## Input formats

The pipeline accepts the following kinds of input:

  * Plain text, with normal Hebrew whitespace behavior. Newlines are assumed to indicate a sentence break, but longer paragraphs will receive automatic sentence splitting too.
  * Gold super-tokenized: if whitespace tokenization is already done, you can leave out `-w`. The system expect one super-token per line in this case (e.g. <bbyt> is on one line)
  * Gold tokenized: if gold morphological segmentation is already done, you can input one gold token per line.
  * XML sentence tags in input: use -s TAGNAME to indicate an XML tag providing gold sentence boundaries.

## Tools used

Currently the following tools are 