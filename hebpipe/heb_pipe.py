#!/usr/bin/python
# -*- coding: utf-8 -*-

import re, sys, io, os, platform
import tempfile
import subprocess
from glob import glob
from diaparser.parsers import Parser
from stanza.utils.conll import CoNLL
from depedit import DepEdit
import stanza
from stanza.models.common.doc import Document
import torch

from rftokenizer import RFTokenizer
try:  # Module usage
    from .lib.xrenner import Xrenner
    from .lib._version import __version__
    from .lib.tt2conll import conllize
    from .lib.append_column import inject_col
    from .lib.sent_split import toks_to_sents
    from .lib.whitespace_tokenize import add_space_after, tokenize as whitespace_tokenize
    from .lib.mtlmodel import Tagger
except ImportError:  # direct script usage
    from lib.xrenner import Xrenner
    from lib._version import __version__
    from lib.tt2conll import conllize
    from lib.append_column import inject_col
    from lib.sent_split import toks_to_sents
    from lib.whitespace_tokenize import add_space_after, tokenize as whitespace_tokenize
    from lib.mtlmodel import Tagger

PY3 = sys.version_info[0] > 2

if PY3:
    from urllib.request import urlretrieve
else:
    from urllib import urlretrieve

inp = input if PY3 else raw_input

script_dir = os.path.dirname(os.path.realpath(__file__))
lib_dir = script_dir + os.sep + "lib" + os.sep
bin_dir = script_dir + os.sep + "bin" + os.sep
data_dir = script_dir + os.sep + "data" + os.sep
model_dir = script_dir + os.sep + "models" + os.sep
marmot_path = bin_dir + "Marmot" + os.sep

KNOWN_PUNCT = {'’','“','”'}  # Hardwired tokens to tag as punctutation (unicode glyphs not in training data)
morph_deped = DepEdit(config_file=lib_dir+"partial_morph.ini")
tags = {"NOUN", "VERB", "ADJ", "ADV", "PROPN"}
lex_data = open(data_dir + "heb.lemma", encoding="utf8").readlines()
lex = {}
for l in lex_data:
    word, tag, lemma = l[:-1].split("\t")
    if tag in tags:
        lex["\t".join([word,tag])] = lemma

binyan_data = open(data_dir + "heb.binyan", encoding="utf8").readlines()
binyan_lookup = {}
binyan_lemma_lookup = {}
for l in binyan_data:
    word, lemma, binyan = l[:-1].split("\t")
    binyan_lookup[(word,lemma)] = binyan
    binyan_lemma_lookup[lemma] = binyan


def init_lemmatizer(cpu=False, no_post_process=False):
    lemmatizer = stanza.Pipeline("he", package="htb", processors="lemma", tokenize_no_ssplit=True,
                         tokenize_pretokenized=True,
                          lemma_pretagged=True,
                          depparse_pretagged=True,
                          use_gpu=not cpu,
                          lemma_model_path=model_dir + "stanza" + os.sep + "he_lemmatizer.pt")
    lemmatizer.do_post_process = False if no_post_process else True
    return lemmatizer


def log_tasks(opts):
    sys.stderr.write("\nRunning tasks:\n" +"="*20 + "\n")
    if opts.sent not in ["auto","none"]:
        sys.stderr.write("o Splitting sentences based on tag: "+opts.sent+"\n")
    elif opts.sent == "auto":
        if opts.punct_sentencer:
            sys.stderr.write("o Automatic sentence splitting (punctuation based)\n")
        else:
            sys.stderr.write("o Automatic sentence splitting (neural)\n")
    if opts.whitespace:
        sys.stderr.write("o Whitespace tokenization\n")
    if opts.tokenize:
        sys.stderr.write("o Morphological segmentation\n")
    if opts.posmorph:
        sys.stderr.write("o POS and Morphological tagging\n")
    if opts.lemma:
        sys.stderr.write("o Lemmatization\n")
    if opts.dependencies:
        sys.stderr.write("o Dependency parsing\n")
    if opts.entities:
        sys.stderr.write("o Entity recognition\n")
    if opts.coref:
        sys.stderr.write("o Coreference resolution\n")

    sys.stderr.write("\n")


def diagnose_opts(opts):
    if not opts.posmorph and not opts.whitespace and not opts.tokenize and not opts.lemma \
        and not opts.dependencies and not opts.entities and not opts.coref:
        if not opts.quiet:
            sys.stderr.write("! You selected no processing options\n")
            sys.stderr.write("! Assuming you want all processing steps\n")
        opts.whitespace = True
        opts.tokenize = True
        opts.posmorph = True
        opts.lemma = True
        opts.dependencies = True
        opts.entities = True
        opts.coref = True

    added = []
    trigger = ""
    if opts.dependencies:
        trigger = "depenedencies"
        if not opts.posmorph:
            added.append("posmorph")
            opts.posmorph = True
        if not opts.lemma:
            added.append("lemma")
            opts.lemma = True

    if len(added)>0:
        sys.stderr.write("! You selected "+trigger+"\n")
        sys.stderr.write("! Turning on options: "+",".join(added) +"\n")

    added = []
    if opts.whitespace:
        trigger = "whitespace tokenization"
        if not opts.tokenize:
            added.append("tokenize")
            opts.tokenize = True
    if len(added)>0:
        sys.stderr.write("! You selected "+trigger+"\n")
        sys.stderr.write("! Turning on options: "+",".join(added) +"\n")
    return opts


def groupify(output,anno):
    groups = ""
    current_group = ""
    for line in output.split("\n"):
        if " "+anno+"=" in line:
            current_group += re.search(anno + r'="([^"]*)"',line).group(1)
        if line.startswith("</") and "_group" in line:
            groups += current_group +"\n"
            current_group = ""

    return groups


def get_bound_group_map(data):

    mapping = {}
    data = data.split("\n")
    # Ignore markup
    data = [u for u in data if not (u.startswith("<") and u.endswith(">"))]
    counter = 0
    for i, line in enumerate(data):
        super_token = line.replace("|","") if line != "|" else "|"
        segs = line.split("|") if line != "|" else ["|"]
        for j, seg in enumerate(segs):
            if len(segs)>1 and j == 0:
                mapping[counter] = (super_token,len(segs))
                super_token = ""
            counter += 1

    return mapping


def remove_nesting_attr(data, nester, nested, attr="xml:lang"):
    """
    Removes attribute on nesting element if a nested element includes it

    :param data: SGML input
    :param nester: nesting tag, e.g. "norm"
    :param nested: nested tag, e.g. "morph"
    :param attr: attribute, e.g. "lang"
    :return: cleaned SGML
    """

    if attr not in data:
        return data
    flagged = []
    in_attr_nester = False
    last_nester = -1
    lines = data.split("\n")
    for i, line in enumerate(lines):
        if nester + "=" in line and attr+"=" in line:
            in_attr_nester = True
            last_nester = i
        if "</" + nester + ">" in line:
            in_attr_nester = False
        if nested in line and attr+"=" in line and in_attr_nester and last_nester > -1:
            flagged.append(last_nester)
            in_attr_nester = False
    for i in flagged:
        lines[i] = re.sub(' '+attr+'="[^"]+"','',lines[i])
    return "\n".join(lines)


def tok_from_norm(data):
    """
    Takes TT-SGML, extracts norm attribute, and replaces existing tokens with norm values while retaining SGML tags.
    Used to feed parser norms while retaining SGML sentence separators.

    :param data: TTSGML with <norm norm=...> and raw tokens to replace
    :return: TTSGML with tags preserved and tokens replace by norm attribute values
    """

    outdata = []
    norm = ""
    for line in data.replace("\r","").split("\n"):
        if line.startswith("<"):
            m = re.search(r'norm="([^"]*)"',line)
            if m is not None:
                norm = m.group(1)
            outdata.append(line)
        else:
            if norm != "":
                outdata.append(norm)
                norm=""
    return "\n".join(outdata) + "\n"


def read_attributes(input,attribute_name):
    out_stream =""
    for line in input.split('\n'):
        if attribute_name + '="' in line:
            m = re.search(attribute_name+r'="([^"]*)"',line)
            if m is None:
                print("ERR: cant find " + attribute_name + " in line: " + line)
                attribute_value = ""
            else:
                attribute_value = m.group(1)
            if len(attribute_value)==0:
                attribute_value = "_warn:empty_"+attribute_name+"_"
            out_stream += attribute_value +"\n"
    return out_stream


def merge_into_tag(tag_to_kill, tag_to_merge_into,stream):
    vals = []
    cleaned_stream = ""
    for line in stream.split("\n"):
        if " "+tag_to_kill + "=" in line:
            val = re.search(" " + tag_to_kill+'="([^"]*)"',line).group(1)
            vals.append(val)
        elif "</" + tag_to_kill + ">" in line:
            pass
        else:
            cleaned_stream += line + "\n"
    injected = inject(tag_to_kill,"\n".join(vals).strip(),tag_to_merge_into,cleaned_stream)
    return injected


def inject_conllu(source_conllu, target_conllu, columns=None):
    if columns is None:
        columns = [6,7]
    lines = source_conllu.split("\n")
    insertions = []
    for line in lines:
        if "\t" in line:
            fields = line.split("\t")
            insertion = []
            for colnum in columns:
                insertion.append(fields[colnum])
            insertions.append(insertion)
    output = []
    toknum = 0
    for line in target_conllu.split("\n"):
        if "\t" in line:
            fields = line.split("\t")
            if "-" not in fields[0]:
                insertion = insertions[toknum]
                for i, val in enumerate(insertion):
                    fields[columns[i]] = val
                line = "\t".join(fields)
                toknum += 1
        output.append(line)
    return "\n".join(output)


def get_col(data, colnum):
    if not isinstance(data,list):
        data = data.split("\n")

    splits = [row.split("\t") for row in data if "\t" in row]
    return [r[colnum] for r in splits]


def exec_via_temp(input_text, command_params, workdir="", outfile=False):
    temp = tempfile.NamedTemporaryFile(delete=False)
    if outfile:
        temp2 = tempfile.NamedTemporaryFile(delete=False)
    output = ""
    try:
        temp.write(input_text.encode("utf8"))
        temp.close()

        if outfile:
            command_params = [x if 'tempfilename2' not in x else x.replace("tempfilename2",temp2.name) for x in command_params]
        command_params = [x if 'tempfilename' not in x else x.replace("tempfilename",temp.name) for x in command_params]
        if workdir == "":
            proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            (stdout, stderr) = proc.communicate()
        else:
            proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE,cwd=workdir)
            (stdout, stderr) = proc.communicate()
        if outfile:
            if PY3:
                output = io.open(temp2.name,encoding="utf8").read()
            else:
                output = open(temp2.name).read()
            temp2.close()
            os.remove(temp2.name)
        else:
            output = stdout
        #print(stderr)
        proc.terminate()
    except Exception as e:
        print(e)
    finally:
        os.remove(temp.name)
        return output


def exec_via_temp_old(input_text, command_params, workdir=""):
    temp = tempfile.NamedTemporaryFile(delete=False)
    exec_out = ""
    try:
        if PY3:
            temp.write(input_text.encode("utf8"))
        else:
            temp.write(input_text.encode("utf8"))
        temp.close()

        command_params = [x if x != 'tempfilename' else temp.name for x in command_params]
        if workdir == "":
            proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE)
            (stdout, stderr) = proc.communicate()
        else:
            proc = subprocess.Popen(command_params, stdout=subprocess.PIPE,stdin=subprocess.PIPE,stderr=subprocess.PIPE,cwd=workdir)
            (stdout, stderr) = proc.communicate()

        exec_out = stdout
    except Exception as e:
        print(e)
    finally:
        os.remove(temp.name)
        #if PY3:
        exec_out = exec_out.decode("utf8")
        return exec_out


def get_origs(data):
    origs = []
    current = ""
    for line in data.split("\n"):
        if "</norm>" in line:
            origs.append(current)
            current = ""
        if not line.startswith("<"):  # Token line
            current += line

    return "\n".join(origs)


def inject(attribute_name, contents, at_attribute,into_stream,replace=True):
    insertions = contents.split('\n')
    injected = ""
    i=0
    for line in into_stream.split("\n"):
        if at_attribute + "=" in line:
            if i >= len(insertions):
                raise Exception("Error out of bounds at element " + str(i) + " in document beginning " + into_stream[:1000])
            if len(insertions[i])>0:
                if at_attribute == attribute_name:  # Replace old value of attribute with new one
                    line = re.sub(attribute_name+'="[^"]+"',attribute_name+'="'+insertions[i]+'"',line)
                else:  # Place before specific at_attribute
                    if replace or " " + attribute_name + "=" not in line:
                        line = re.sub(at_attribute+"=",attribute_name+'="'+insertions[i]+'" '+at_attribute+"=",line)
            i += 1
        injected += line + "\n"
    return injected


def extract_conll(conll_string):
    conll_string = conll_string.replace("\r","").strip()
    sentences = conll_string.split("\n\n")
    ids = ""
    funcs = ""
    parents = ""
    id_counter = 0
    offset = 0
    for sentence in sentences:
        tokens = sentence.split("\n")
        for token in tokens:
            if "\t" in token:
                id_counter +=1
                ids += "u"+ str(id_counter) + "\n"
                cols = token.split("\t")
                funcs += cols[7].replace("ROOT","root") +"\n"
                if cols[6] == "0":
                    parents += "#u0\n"
                else:
                    parents += "#u" + str(int(cols[6])+offset)+"\n"
        offset = id_counter
    return ids, funcs, parents


def inject_tags(in_sgml,insertion_specs,around_tag="norm",inserted_tag="mwe"):
    """

    :param in_sgml: input SGML stream including tags to surround with new tags
    :param insertion_specs: list of triples (start, end, value)
    :param around_tag: tag of span to surround by insertion
    :return: modified SGML stream
    """
    if len(insertion_specs) == 0:
        return in_sgml

    counter = -1
    next_insert = insertion_specs[0]
    outlines = []
    for line in in_sgml.split("\n"):
        if line.startswith("<" + around_tag + " "):
            counter += 1
            if next_insert[0] == counter:  # beginning of a span
                outlines.append("<" + inserted_tag + " " + inserted_tag + '="' + next_insert[2] + '">')
        outlines.append(line)
        if line.startswith("</" + around_tag + ">"):
            if next_insert[1] == counter:  # end of a span
                outlines.append("</" + inserted_tag + ">")

    return "\n".join(outlines)


def lemmatize(lemmatizer, conllu, morphs):
    def clean_final(text):
        finals = {"פ":"ף","כ":"ך","מ":"ם","נ":"ן","צ":"ץ"}
        if text[-1] in finals:
            text = text[:-1] + finals[text[-1]]
        return text

    def post_process(word, pos, lemma, morph):
        if word == lemma:
            if word + "\t" + pos in lex:
                if pos == "VERB" and "Fut" in morph:
                    lemma = lex[word + "\t" + pos]
                if pos == "VERB" and "Pres" in morph:
                    lemma = lex[word + "\t" + pos]
                if pos == "VERB" and "Part" in morph:
                    lemma = lex[word + "\t" + pos]
                if pos in ["NOUN", "ADJ"] and "Plur" in morph:
                    lemma = lex[word + "\t" + pos]
            else:
                if "Plur" in morph and pos in ["NOUN", "ADJ"] and (
                        word.endswith("ים") or word.endswith("ות")):
                    lemma = lemma[:-2]
                    if word.endswith("ות"):
                        lemma += "ה"
                    lemma = clean_final(lemma)
        return lemma

    uposed = [[l.split("\t") for l in s.split("\n")] for s in conllu.strip().split("\n\n")]
    dicts = CoNLL.convert_conll(uposed)
    for sent in dicts:
        for tok in sent:
            tok["id"] = int(tok["id"][0])
    doc = Document(dicts)
    lemmatized = lemmatizer(doc)
    output = []
    counter = 0
    for sent in lemmatized.sentences:
        for tok in sent.tokens:
            word = tok.words[0]
            lemma = word.lemma
            if lemmatizer.do_post_process:
                lemma = post_process(word.text, word.upos, word.lemma, morphs[counter])
            row = [str(word.id), word.text, lemma, word.upos, word.xpos, '_', str(word.head), "_", "_", "_"]
            output.append("\t".join(row))
            counter += 1
        output.append("")
    lemmatized = "\n".join(output)
    lemmatized = get_col(lemmatized,2)

    return lemmatized


def diaparse(parser, conllu):
    sents = conllu.strip().split("\n\n")
    parser_input = []

    for sent in sents:
        words = [(l.split("\t")[0],l.split("\t")[1]) for l in sent.split("\n") if "\t" in l]
        words = [w[1] for w in words if "." not in w[0] and "-" not in w[0]]
        parser_input.append(words)

    dataset = parser.predict(parser_input, prob=True)
    torch.cuda.empty_cache()

    out_parses = []
    for sent in dataset.sentences:
        out_parses.append(str(sent).strip())

    merged = inject_conllu("\n\n".join(out_parses) + "\n\n", conllu)

    return merged


def postprocess_morph(feats, words, lemmas, tags):
    def add_feat(morph, feat):
        if morph == "_":
            return feat
        else:
            attrs = morph.split("|")
            attrs = {a.split("=")[0]:a.split("=")[1] for a in attrs}
            attrs[feat.split("=")[0]] = feat.split("=")[1]
            return "|".join(sorted([a + "=" + attrs[a] for a in attrs]))

    output = []
    for i, lemma in enumerate(lemmas):
        word = words[i]
        feat = feats[i]
        tag = tags[i]
        if "HebBinyan" in feat or tag == "VERB":  # Rely on BERT to notice that binyan is needed, or if it's a VERB
            if (word,lemma) in binyan_lookup:
                feat = add_feat(feat,"HebBinyan=" + binyan_lookup[(word,lemma)])
            elif lemma in binyan_lemma_lookup:
                feat = add_feat(feat,"HebBinyan=" + binyan_lemma_lookup[lemma])
        output.append(feat)
    return output


def check_requirements():
    models_OK = True
    model_files = ["heb.sm" + str(sys.version_info[0]), "heb.xrm",
                   "heb.diaparser", "heb.sbdposmorph.pt",
                   "stanza" + os.sep + "he_lemmatizer.pt",
                   "stanza" + os.sep + "he_htb.pretrain.pt",
                   ]
    for model_file in model_files:
        if not os.path.exists(model_dir + model_file):
            sys.stderr.write("! Model file " + model_file + " missing in ./models/\n")
            models_OK = False

    return models_OK


def download_requirements(models_ok=True):
    urls = []
    if not models_ok:
        models_base = "http://gucorpling.org/amir/download/heb_models_v3/"
        urls.append(models_base + "heb.sm" + str(sys.version_info[0]))
        urls.append(models_base + "heb.diaparser")
        urls.append(models_base + "heb.xrm")
        urls.append(models_base + "he_htb.pretrain.pt")
        urls.append(models_base + "he_lemmatizer.pt")
        urls.append(models_base + 'heb.sbdposmorph.pt')
    for u in urls:
        sys.stderr.write("o Downloading from " + str(u) + "\n")
        base_name = u[u.rfind("/") + 1:]
        if "he_" in base_name:
            urlretrieve(u, model_dir + "stanza" + os.sep + base_name)
        else:
            urlretrieve(u, model_dir + base_name)
    sys.stderr.write("\n")


def nlp(input_data, do_whitespace=True, do_tok=True, do_tag=True, do_lemma=True, do_parse=True, do_entity=True,
        out_mode="conllu", sent_tag=None, preloaded=None,  from_pipes=False,cpu=False):

    data = input_data.replace("\t","")
    data = data.replace("\r","")

    if from_pipes:
        input_data = input_data.replace("|","")

    if preloaded is not None:
        rf_tok, xrenner, mtltagger,parser, lemmatizer = preloaded
    else:
        rf_tok = RFTokenizer(model=model_dir + "heb.sm" + str(sys.version_info[0]))
        xrenner = Xrenner(model=model_dir + "heb.xrm")
        parser = None if not do_parse else Parser.load(model_dir + "heb.diaparser",verbose=False)
        lemmatizer = None if not do_lemma and not do_tag else init_lemmatizer()
        mtltagger = Tagger(trainflag=False, bestmodelpath=model_dir, sequencelength=256,cpu=cpu)

    if do_whitespace:
        data = whitespace_tokenize(data, abbr=data_dir + "heb_abbr.tab",add_sents=sent_tag=="auto", from_pipes=from_pipes)

    if from_pipes:
        tokenized = data
    else:
        if do_tok:
            tokenized = rf_tok.rf_tokenize(data.strip().split("\n"))
            tokenized = "\n".join(tokenized)
        else:
            # Assume data is already one token per line
            tokenized = data

    bound_group_map = get_bound_group_map(tokenized) if out_mode == "conllu" else None

    if mtltagger:
        tagged_conllu, tokenized, morphs, words = mtltagger.predict(tokenized,sent_tag=sent_tag,checkpointfile=model_dir + 'heb.sbdposmorph.pt')

    if out_mode == "pipes":
        return tokenized
    else:
        tokenized = tokenized.split("\n")
        retokenized = []
        for line in tokenized:
            if line == '|':
                retokenized.append(line)
            else:
                retokenized.append("\n".join(line.split("|")))
        tokenized =  "\n".join(retokenized)

    if sent_tag == 'auto': sent_tag = 's'

    del mtltagger
    del rf_tok
    torch.cuda.empty_cache()

    if do_tag:
        zeros = ["0" for i in range(len(morphs))]
        zero_conllu = inject_col(zeros, tagged_conllu, into_col=6, skip_supertoks=True)
        lemmas = lemmatize(lemmatizer, zero_conllu, morphs)
        tagged = inject_col(tagged_conllu, tokenized, 4)

        if do_lemma:
            lemmatized = inject_col(lemmas, tagged, -1)
        else:
            lemmatized = tagged

        tags = [l.split("\t")[3] for l in tagged_conllu.split("\n") if "\t" in l]
        morphs = postprocess_morph(morphs, words, lemmas, tags)
        morphed = inject_col(morphs, lemmatized, -1)

        if not do_parse:
            if out_mode == "conllu":
                conllized = conllize(morphed, tag="PUNCT", element=sent_tag, no_zero=True, super_mapping=bound_group_map,
                                     attrs_as_comments=True)
                conllized = add_space_after(input_data,conllized)
                return conllized
            else:
                if not PY3:
                    morphed = morphed.decode("utf8")
                return morphed

    else:
        if out_mode == "conllu":
            conllized = conllize(tokenized, tag="PUNCT", element=sent_tag, no_zero=True, super_mapping=bound_group_map,
                                 attrs_as_comments=True)
            conllized = add_space_after(input_data,conllized)
            return conllized
        else:
            return tokenized

    if do_parse:
        conllized = conllize(morphed,tag="PUNCT",element=sent_tag,no_zero=True,super_mapping=bound_group_map,attrs_as_comments=True,ten_cols=True)
        parsed = diaparse(parser, conllized)
        parsed = morph_deped.run_depedit(parsed)

        if do_entity:
            xrenner.docname = "_"
            ents = xrenner.analyze(parsed,"conll_sent")
            ents = get_col(ents, -1)
            entified = inject_col(ents, parsed, col=-1, into_col=9, skip_supertoks=True)
            entified = add_space_after(input_data,entified)
            if PY3:
                return entified
            else:
                return entified.decode("utf8")
        else:
            parsed = add_space_after(input_data,parsed)
            return parsed
    else:
        if out_mode == "conllu":
            conllized = conllize(tagged,tag="PUNCT",element=sent_tag,no_zero=True,super_mapping=bound_group_map,attrs_as_comments=True)
            conllized = add_space_after(input_data,conllized)
            return conllized
        else:
            return tagged


def run_hebpipe():

    if sys.version_info[0] == 2 and sys.version_info[1] < 7:
        sys.stderr.write("Python versions below 2.7 are not supported.\n")
        sys.stderr.write("Your Python version:\n")
        sys.stderr.write(".".join([str(v) for v in sys.version_info[:3]]) + "\n")
        sys.exit(0)

    from argparse import ArgumentParser, RawDescriptionHelpFormatter

    parser = ArgumentParser(formatter_class=RawDescriptionHelpFormatter)
    parser.prog = "HebPipe - NLP Pipeline for Hebrew"
    parser.usage = "python heb_pipe.py [OPTIONS] files"
    parser.epilog = """Example usage:
--------------
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
"""
    parser.add_argument("files", help="File name or pattern of files to process (e.g. *.txt)")

    g1 = parser.add_argument_group("standard module options")
    g1.add_argument("-w","--whitespace", action="store_true", help='Perform white-space based tokenization of large word forms')
    g1.add_argument("-t","--tokenize", action="store_true", help='Tokenize large word forms into smaller morphological segments')
    g1.add_argument("-p","--posmorph", action="store_true", help='Do POS and Morph tagging')
    g1.add_argument("-l","--lemma", action="store_true", help='Do lemmatization')
    g1.add_argument("-d","--dependencies", action="store_true", help='Parse with dependency parser')
    g1.add_argument("-e","--entities", action="store_true", help='Add entity spans and types')
    g1.add_argument("-c","--coref", action="store_true", help='Add coreference annotations')
    g1.add_argument("-s","--sent", action="store", default="auto", help='XML tag to split sentences, e.g. sent for <sent ..> or none for no splitting (otherwise automatic sentence splitting)')
    g1.add_argument("-o","--out", action="store", choices=["pipes","conllu","sgml"], default="conllu", help='Output CoNLL format, SGML or just tokenize with pipes')

    g2 = parser.add_argument_group("less common options")
    g2.add_argument("-q","--quiet", action="store_true", help='Suppress verbose messages')
    g2.add_argument("-x","--extension", action="store", default='conllu', help='Extension for output files (default: .conllu)')
    g2.add_argument("--cpu", action="store_true", help='Use CPU instead of GPU (slower)')
    g2.add_argument("--disable_lex", action="store_true", help='Do not use lexicon during lemmatization')
    g2.add_argument("--dirout", action="store", default=".", help='Optional output directory (default: this dir)')
    g2.add_argument("--punct_sentencer", action="store_true", help='Only use punctuation (.?!) to split sentences (deprecated)')
    g2.add_argument("--from_pipes", action="store_true", help='Input contains subtoken segmentation with the pipe character (no automatic tokenization is performed)')
    g2.add_argument("--version", action="store_true", help='Print version number and quit')

    if "--version" in sys.argv:
        sys.stdout.write("HebPipe V" + __version__)
        sys.exit(1)

    opts = parser.parse_args()
    opts = diagnose_opts(opts)

    if opts.cpu:
        torch.cuda.is_available = lambda: False

    dotok = opts.tokenize

    if not opts.quiet:
        try:
            from .lib import timing
        except ImportError:  # direct script usage
            from lib import timing

    files = glob(opts.files)

    if not opts.quiet:
        log_tasks(opts)

    # Check if models are available
    if opts.posmorph or opts.lemma or opts.dependencies or opts.tokenize or opts.entities:
        models_OK = check_requirements()
        if not models_OK:
            sys.stderr.write("! You are missing required software:\n")
            if (opts.posmorph or opts.lemma):
                sys.stderr.write(" - Tagging, lemmatization and morphological analysis require models\n")
            if not models_OK:
                sys.stderr.write(" - Model files in models/ are missing\n")
            response = inp("Attempt to download missing files? [Y/N]\n")
            if response.upper().strip() == "Y":
                download_requirements(models_OK)
            else:
                sys.stderr.write("Aborting\n")
                sys.exit(0)

        lemmatizer = init_lemmatizer(cpu=opts.cpu, no_post_process=opts.disable_lex)
    else:
        lemmatizer = None

    if opts.posmorph:
        mtltagger = Tagger(trainflag=False, bestmodelpath=model_dir, sequencelength=256,cpu=opts.cpu)
    else:
        mtltagger = None

    if dotok:  # Pre-load stacked tokenizer for entire batch
        rf_tok = RFTokenizer(model=model_dir + "heb.sm" + str(sys.version_info[0]))
    else:
        rf_tok = None
    if opts.entities:  # Pre-load stacked tokenizer for entire batch
        xrenner = Xrenner(model=model_dir + "heb.xrm")
    else:
        xrenner = None

    dep_parser = Parser.load(model_dir+"heb.diaparser") if opts.dependencies else None

    for infile in files:
        base = os.path.basename(infile)
        if infile.endswith("." + opts.extension):
            outfile = base.replace("." + opts.extension,".out." + opts.extension)
        elif len(infile) > 4 and infile[-4] == ".":
            outfile = base[:-4] + "." + opts.extension
        else:
            outfile = base + "." + opts.extension

        if not opts.quiet:
            sys.stderr.write("Processing " + base + "\n")

        try:
            input_text = io.open(infile,encoding="utf8").read().replace("\r","")
        except UnicodeDecodeError:  # Fallback to support Windows Hebrew encoding
            input_text = io.open(infile,encoding="cp1255").read().replace("\r","")

        processed = nlp(input_text, do_whitespace=opts.whitespace, do_tok=dotok, do_tag=opts.posmorph, do_lemma=opts.lemma,
                               do_parse=opts.dependencies, do_entity=opts.entities, out_mode=opts.out,
                               sent_tag=opts.sent, preloaded=(rf_tok,xrenner,mtltagger,dep_parser,lemmatizer),
                                from_pipes=opts.from_pipes,cpu=opts.cpu)

        if len(files) > 1:
            with io.open(opts.dirout + os.sep + outfile, 'w', encoding="utf8", newline="\n") as f:
                if not PY3:
                    processed = unicode(processed)
                f.write((processed.strip() + "\n"))
        else:  # Single file, print to stdout
            if PY3:
                sys.stdout.buffer.write((processed+"\n\n").encode("utf8"))
            else:
                print(processed.encode("utf8"))

    fileword = " files\n\n" if len(files) > 1 else " file\n\n"
    sys.stderr.write("\nFinished processing " + str(len(files)) + fileword)


if __name__ == "__main__":
    import logging
    logging.disable(logging.INFO)
    run_hebpipe()

