import pandas as pd
import regex as re
import numpy as np
import nltk
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import pickle
# nltk.download('words')
from nltk.corpus import words
dictwords = set(x.lower() for x in words.words())

'''
https://gitlab.lrz.de/social-rom/datasets/organic-dataset/curated-source-dataset/-/blob/master/english/Data_curation_language.ipynb
'''

# importing contractions compiled bt Ahmed
contractions = {
    "aren't": 'are not',
    "can't": 'cannot',
    "could've": 'could have',
    "couldn't": 'could not',
    "didn't": 'did not',
    "doesn't": 'does not',
    "don't": 'do not',
    'gonna': 'going to',
    'gotta': 'got to',
    "hadn't": 'had not',
    "hasn't": 'has not',
    "haven't": 'have not',
    "he'd": 'he would',
    "he'll": 'he will',
    "he's": 'he is',
    "how's": 'how is',
    "I'd": 'I would',
    "I'll": 'I will',
    "I'm": 'I am',
    "i'm": 'I am',
    "I've": 'I have',
    "isn't": 'is not',
    "it'd": 'it would',
    "it'll": 'it will',
    "it's": 'it is',
    "mayn't": 'may not',
    "may've": 'may have',
    "mightn't": 'might not',
    "might've": 'might have',
    "mustn't": 'must not',
    "must've": 'must have',
    "needn't": 'need not',
    "o'clock": 'of the clock',
    "oughtn't": 'ought not',
    "she'd": 'she would',
    "she'll": 'she will',
    "she's": 'she is',
    "should've": 'should have',
    "shouldn't": 'should not',
    "that's": 'that is',
    "there're": 'there are',
    "there's": 'there is',
    "these're": 'these are',
    "they'd": 'they would',
    "they'll": 'they will',
    "they're": 'they are',
    "they've": 'they have',
    "this's": 'this is',
    "those're": 'those are',
    "wasn't": 'was not',
    "we'd": 'we would',
    "we'll": 'we will',
    "we're": 'we are',
    "we've": 'we have',
    "weren't": 'were not',
    "what'll": 'what will',
    "what're": 'what are',
    "what's": 'what is',
    "what've": 'what have',
    "when's": 'when is',
    "where're": 'where are',
    "where's": 'where is',
    "which's": 'which is',
    "who'd": 'who would',
    "who'll": 'who will',
    "who're": 'who are',
    "who's": 'who is',
    "why's": 'why is',
    "won't": 'will not',
    "would've": 'would have',
    "wouldn't": 'would not',
    "you'd": 'you would',
    "you'll": 'you will',
    "you're": 'you are',
    "you've": 'you have'
}

# replacement dictionary compiled by Ahmed
replacement_dict = {
    '\n' : ' ',
    '!=' : ' not equal ',
    '=' : ' equal ',
    b'\xc2\xae'.decode() : ' registered_sign ', # ®
    '\x92' : ' ',
    '\x91' : ' ',
    '\x96' : ' ',
    b'\xe2\x84\xa2'.decode() : ' trademark_sign ', # ™
    b'\xe2\x80\x90'.decode() : '-',
    '}': ')',
    '{': '(',
    b'\xc2\xb2'.decode() : ' squared ',
    b'\xc2\xa7'.decode() : ' section ',  # §
    b'\xc2\xb0'.decode() : ' degrees ',
    b'\xe2\x80\xa6'.decode() : ' . ',   # …
    '\$' : ' dollar ',
    b'\xe2\x82\xac'.decode() : ' euro ',
    '\|' : ' , ',    
    b'\xc2\xab'.decode() : ' \" ',
    b'\xc2\xbb'.decode() : ' \" ',
    '\+' : ' plus ',
    b'\xc2\xa2'.decode() : ' , ', # ¢
    b'\xe2\x80\x8b'.decode() : ' ',
    '\|' : ',',
    b'\xe2\x80\x93'.decode() : '-', # the long dash  –
    b'\xe2\x80\x94'.decode() : '-', # another long dash
    '\[' : '(',
    '\]' : ')', 
    '&' : ' and ',
    b'\xe2\x80\x9c'.decode() : '\"',
    b'\xe2\x80\x9d'.decode() : '\"',
    b'\xc2\xbd'.decode() : ' half ',
    b'\xc2\xbc'.decode() : ' quarter ',
    b'\xe2\x80\x99'.decode() : '\'',
    b'\xe2\x80\x98'.decode() : '\'',
    b'\xc2\xb4'.decode() : '\'',
    b'\xc2\xb5g'.decode() : ' microgram ',
    '.' : ' . ',
    ',' : ' , '
}

# social_terms_organic_relevant_terms dictionary compiled by Ahmed
social_terms_organic_relevant_terms = {
    'btw' : ' by the way ',
    'tl;dr': ' summary ',
    'tbsp': ' table spoon ',
    'imho': ' in my opinion ',
    'imo' : ' in my opinion ',
    'oganic': ' organic ',
    'orgainc' : ' organic ',
    'tsp': ' tea spoon ',
    'faqs': ' frequently asked questions ',
    'fyi': ' for your information ',
    'pestdicides': ' pesticides ',
    'pestdicide': ' pesticide ',
    'pesiticides': ' pesticides ',
    'ogranic': ' organic ',
    'pestecides': ' pesticides ',
    'nonorganic': ' non organic ',
    'pestcides':' pesticides ',
    '<3': ' love ',
    ' alot ': ' a lot ',
    'thier': ' their ',
    'breastmilk': ' breast milk ',
    'agribusinesses' : ' agricultural businesses ',
    '<a href equal \"': ' ',
    'café': 'cafe'
}

def IsEnglish_dict(y):
    '''identifying english words from data corpus'''
    counter = 0
    for z in y:
        if z in dictwords:
            counter += 1
    return counter

from langdetect import detect_langs

def EnglishOrGermanOrHindi(string):
    '''identifying language using py-library'''
    try:
        res = detect_langs(string)
        lang_id = set(' '.join(re.split(r':', str(item))[0] for item in res).split())
        if 'id' in lang_id:
            return 'id'
        elif 'en' in lang_id:
            return 'en'
        elif 'de' in lang_id:
            return 'de'
        return None
    except:
        return None

'''
code example from `detect_langs`
```python
>>> detect_langs("Otec matka syn.")
[sk:0.572770823327, pl:0.292872522702, cs:0.134356653968]
```
'''

def filter_articles_pandas(maindf):
    '''
    cleaning and filtering articles when processing pandas df
    modified from filter_comments_pandas
    '''
    #Reading non empty articles
    filterdf = pd.DataFrame(maindf[(maindf['article']!='')]['article'])     
    
    #Adding new columns for filter computation
    filterdf['tidy_article'] = filterdf['split_article'] = filterdf['no_words'] = filterdf['no_English_count'] \
                                  = filterdf['ratio'] = filterdf['is_English_dict'] = filterdf['language']  = filterdf['is_English_py_pkg'] \
                                  = filterdf['is_German_py_pkg'] = filterdf['is_Hindi_py_pkg'] = ''
        
    #Replacing multiple punctuations with single
    filterdf.replace({ r'\A\s+|\s+\Z': '', '\n' : '. ', r'\.+': '.', r'\,+': ',', r'\-+': '-', 
                       r"\'+": "'", r'\!+': '!', r'\?+': '?', r'\^+': '^', r'\#+': '#'}, regex=True, inplace=True)
    
    print('removing Devanagari...')
    #Replacing Devanagari(Hindi scripts) with white space
    # filterdf['article'] = [(re.sub('\  +', ' ', (' '.join([(' ' if bool(re.search(r'\p{Devanagari}',y)) else y.strip()) for y in x.split()]).strip() if bool(re.search(r'\p{Devanagari}+',x)) else x.strip()))) for x in filterdf['article']]
    filterdf['article'] = [re.sub(r'\p{Devanagari}', '', y) for y in filterdf['article']]

    print('do some hard-coded replacement...')
    #Replacing contractions compiled by Ahmed
    filterdf['article'] = [' '.join(contractions[p] if p in contractions else (contractions[p.lower()] if p.lower() in contractions else p) for p in y.split()) 
                                    for y in filterdf['article']]

    #social_terms_organic_relevant_terms dictionary compiled by Ahmed
    filterdf['article'] = [' '.join(social_terms_organic_relevant_terms[p.lower()] if p.lower() in social_terms_organic_relevant_terms else p for p in y.split()) 
                                    for y in filterdf['article']]

    #Replacement dictionary compiled by Ahmed
    # filterdf['article'] = [re.sub('\  +', ' ', ' '.join(replacement_dict[p] if p in replacement_dict else p for p in y.split())) for y in filterdf['article']]
    filterdf['article'] = [' '.join(replacement_dict[p] if p in replacement_dict else p for p in y.split()) for y in filterdf['article']]

    print('removing url...')
    #White spacing url links
    # filterdf['tidy_article'] = [re.split(r'http|https|www',x)[0] for x in filterdf['article']]
    filterdf['article'] = [re.sub(r'(http|https|www)\S+', ' ', x) for x in filterdf['article']]
    
    #Filtering only words
    filterdf['tidy_article'] = [re.sub(r"[^\P{P}']+", ' ', x).strip() for x in filterdf['article']]
    
    filterdf['split_article'] = [x.lower().split() for x in filterdf['tidy_article']]
    
    filterdf['no_words'] = [len(x) for x in filterdf['split_article']]
    
    print('calculating filtering criteria...')
    #Checking English words from dictionary
    filterdf['no_English_count'] = [IsEnglish_dict(x) for x in filterdf['split_article']]
    
    filterdf['ratio'] = filterdf['no_English_count'] / filterdf['no_words']
    
    #Count of English dictionary words with 50% cutoff threshold
    filterdf['is_English_dict'] =  [1 if x > 0.5 else 0 for x in filterdf['ratio']]
    
    #Identifyting language using langdetect python library
    filterdf['language'] =  [EnglishOrGermanOrHindi(x) for x in filterdf['article']]
       
    filterdf['is_English_py_pkg'] = [1 if x=='en' else 0 for x in filterdf['language']]
    
    filterdf['is_German_py_pkg'] = [1 if x=='de' else 0 for x in filterdf['language']]
    
    filterdf['is_Hindi_py_pkg'] = [1 if x=='id' else 0 for x in filterdf['language']]
    
    #Again replacing multiple punctuations with single
    filterdf.replace({r'\.+': '.', r'\,+': ',', r'\-+': '-', r"\'+": "'", r'\!+': '!', r'\?+': '?', r'\^+': '^', r'\#+': '#', r'\  +': ' '}, regex=True, inplace=True)    
       
    #Reading final non empty comments (post pre-processing)
    filterdf1 = filterdf[(filterdf['article']!='')]
    
    #Removing only the auto blacklisted comments
    #Then exporting just the 'comments_text' column after removing duplicates
    final_filterdf = filterdf1[~(filterdf1['is_English_dict'].isin(['0']) & filterdf1['is_English_py_pkg'].isin(['0']))][['article']].drop_duplicates()
       
    return final_filterdf

def filter_comments_pandas(maindf):
    '''
    cleaning and filtering comments when processing pandas df
    Note:
        'tidy_comment' is for counting words
        'comment': original version (Data_curation_language.ipynb) didn't remove urls, but here we want to also remove urls
    '''
    #Reading non empty comments
    filterdf = pd.DataFrame(maindf[(maindf['comment']!='')]['comment'])
    print(f'original length: {maindf.shape} \t length after filter out blank: {filterdf.shape}')
    
    #Adding new columns for filter computation
    filterdf['tidy_comment'] = filterdf['split_comment'] = filterdf['no_words'] = filterdf['no_English_count'] \
                                  = filterdf['ratio'] = filterdf['is_English_dict'] = filterdf['language']  = filterdf['is_English_py_pkg'] \
                                  = filterdf['is_German_py_pkg'] = filterdf['is_Hindi_py_pkg'] = ''
        
    #Replacing multiple punctuations with single
    filterdf.replace({ r'\A\s+|\s+\Z': '', '\n' : '. ', r'\.+': '.', r'\,+': ',', r'\-+': '-', 
                       r"\'+": "'", r'\!+': '!', r'\?+': '?', r'\^+': '^', r'\#+': '#'}, regex=True, inplace=True)
    
    print('removing Devanagari...')
    #Replacing Devanagari(Hindi scripts) with white space
    filterdf['comment'] = [(re.sub('\  +', ' ', (' '.join([(' ' if bool(re.search(r'\p{Devanagari}',y)) else y.strip()) for y in x.split()]).strip() if bool(re.search(r'\p{Devanagari}+',x)) else x.strip()))) for x in filterdf['comment']]
    # filterdf['comment'] = [re.sub(r'\p{Devanagari}', '', y) for y in filterdf['comment']]

    print('do some hard-coded replacement...')
    #Replacing contractions compiled by Ahmed
    filterdf['comment'] = [' '.join(contractions[p] if p in contractions else (contractions[p.lower()] if p.lower() in contractions else p) for p in y.split()) 
                                    for y in filterdf['comment']]

    #social_terms_organic_relevant_terms dictionary compiled by Ahmed
    filterdf['comment'] = [' '.join(social_terms_organic_relevant_terms[p.lower()] if p.lower() in social_terms_organic_relevant_terms else p for p in y.split()) 
                                    for y in filterdf['comment']]

    #Replacement dictionary compiled by Ahmed
    # filterdf['article'] = [re.sub('\  +', ' ', ' '.join(replacement_dict[p] if p in replacement_dict else p for p in y.split())) for y in filterdf['comment']]
    filterdf['comment'] = [' '.join(replacement_dict[p] if p in replacement_dict else p for p in y.split()) for y in filterdf['comment']]

    print('removing url...')
    #White spacing url links
    # filterdf['tidy_comment'] = [re.split(r'http|https|www',x)[0] for x in filterdf['comment']]
    filterdf['comment'] = [re.sub(r'(http|https|www)\S+', ' ', x) for x in filterdf['comment']]
    
    #Filtering only words
    filterdf['tidy_comment'] = [re.sub(r"[^\P{P}']+", ' ', x).strip() for x in filterdf['comment']]
    
    filterdf['split_comment'] = [x.lower().split() for x in filterdf['tidy_comment']]
    
    filterdf['no_words'] = [len(x) for x in filterdf['split_comment']]
    
    print('calculating filtering criteria...')
    #Checking English words from dictionary
    filterdf['no_English_count'] = [IsEnglish_dict(x) for x in filterdf['split_comment']]
    
    filterdf['ratio'] = filterdf['no_English_count'] / filterdf['no_words']
    
    #Count of English dictionary words with 50% cutoff threshold
    filterdf['is_English_dict'] =  [1 if x > 0.5 else 0 for x in filterdf['ratio']]
    
    #Identifyting language using langdetect python library
    filterdf['language'] =  [EnglishOrGermanOrHindi(x) for x in filterdf['comment']]
       
    filterdf['is_English_py_pkg'] = [1 if x=='en' else 0 for x in filterdf['language']]
    
    filterdf['is_German_py_pkg'] = [1 if x=='de' else 0 for x in filterdf['language']]
    
    filterdf['is_Hindi_py_pkg'] = [1 if x=='id' else 0 for x in filterdf['language']]
    
    #Again replacing multiple punctuations with single
    filterdf.replace({r'\.+': '.', r'\,+': ',', r'\-+': '-', r"\'+": "'", r'\!+': '!', r'\?+': '?', r'\^+': '^', r'\#+': '#', r'\  +': ' '}, regex=True, inplace=True)    
       
    #Reading final non empty comments (post pre-processing)
    filterdf1 = filterdf[(filterdf['comment']!='')]
    
    #Removing only the auto blacklisted comments
    #Then exporting just the 'comments_text' column after removing duplicates
    final_filterdf = filterdf1[~(filterdf1['is_English_dict'].isin(['0']) & filterdf1['is_English_py_pkg'].isin(['0']))][['comment']].drop_duplicates()
    print(f'final dataframe length: {final_filterdf.shape}')
       
    return final_filterdf

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
    with open(filename, 'wb') as f:
        pickle.dump(clean_corpus, f)

def pickle_load(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)