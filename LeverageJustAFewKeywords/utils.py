import pandas as pd
import regex as re
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
import os
# nltk.download('words')
from nltk.corpus import words
dictwords = set(x.lower() for x in words.words())

def tokenize_sentence(sent):
    tokenizer = RegexpTokenizer(r'\w+')
    tokenized_sent = tokenizer.tokenize(sent)
    return tokenized_sent

def lemmatize_sentence(sent):
    lemmatier = WordNetLemmatizer()
    if isinstance(sent, str):   # raw sentence
        sent = tokenize_sentence(sent)
    tokens = [lemmatier.lemmatize(token) for token in sent]
    return tokens

def remove_wordlist(sent, word_list=None):
    if isinstance(sent, str):
        sent = tokenize_sentence(sent)
    if word_list:
        tokens = [token for token in sent if token not in word_list]
    return tokens

def pickle_save(clean_corpus, filename):
    base_dir = os.path.dirname(filename)
    os.makedirs(base_dir, exist_ok=True)
    with open(filename, 'wb') as f:
        pickle.dump(clean_corpus, f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def list2dict(l, keys=None):
    if not keys:
        keys = np.arange(len(l))
        keys = [str(k) for k in keys]
    d = {}
    for i, k in enumerate(keys):
        d[k] = l[i]
    return d

def read_aspect_name(file_name):
    with open(file_name, 'r') as f:
        line = f.readline()
        line = line.strip().split('|')
    return line

def flatten_dict(d):
    flat_dict = {}
    for key, value in d.items():
        if isinstance(value, dict):
            for k, v in d[key].items():
                flat_dict[k] = v
        else:
            flat_dict[key] = value
    return flat_dict