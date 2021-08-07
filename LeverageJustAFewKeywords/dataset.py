import json
import logging
import gensim
import os
import csv
import numpy as np
import torch
from mittens import GloVe, Mittens
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils import data
from transformers import BertTokenizerFast
import random
from nltk.corpus import stopwords
import pickle
from time import time

from utils import *

logging.basicConfig(level=logging.INFO)
oposum_domains = ['bags_and_cases', 'bluetooth', 'boots', 'keyboards', 'tv', 'vacuums']

class Dataset(data.Dataset):
    def __init__(self, data_base_dir, domain, stu_params, maxlen=128):
        dataset_mode = 'train'
        if domain in oposum_domains:
            aspect_init_file = os.path.join(data_base_dir, f'{domain}.30.txt')
            data_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}.json')
        elif domain == 'organic':   # organic dataset
            aspect_init_file = os.path.join(data_base_dir, 'seedwords_30_coarse.txt')
            data_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}.json')
        else:
            raise Exception(f"There is no such {domain} domain dataset.")
        
        # self.maxlen = maxlen
        # data
        self.data_orig = self.load_data(data_file)

        # embedding
        # logging.info("loading vocab and representing sentences as index ...")
        self.emb_type = stu_params['pretrained']
        if self.emb_type == 'word2vec':
            # load data and process on the fly, but too slow
            # self.vocab = self.build_shift_vocab_word2vec(stu_params['wv_file'])
            # self.data_idx = [self.build_text_index(s, self.vocab) for s in self.data_orig]
            # self.data_length = [len(s) for s in self.data_idx]

            # load data directly from files 
            vocab_file = os.path.join(data_base_dir, f'{domain}_vocab_w2v.txt')
            self.vocab = self.load_vocab_from_file(vocab_file)
            supplement_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}_supplement_w2v.pkl')
            supplement = pickle_load(supplement_file)
            self.data_idx = supplement['data_idx']      # TODO: deal with blank list
            self.data_length = supplement['data_length']

        elif self.emb_type == 'glove':
            # self.vocab = self.build_shift_vocab_glove(stu_params['wv_file'])
            # self.data_idx = [self.build_text_index(s, self.vocab) for s in self.data_orig]
            # self.data_length = [len(s) for s in self.data_idx]
            vocab_file = os.path.join(data_base_dir, f'{domain}_vocab_glove.txt')
            self.vocab = self.load_vocab_from_file(vocab_file)
            supplement_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}_supplement_glove.pkl')
            supplement = pickle_load(supplement_file)
            self.data_idx = supplement['data_idx']  # TODO: deal with blank list
            self.data_length = supplement['data_length']


        elif self.emb_type == 'bert-base-uncased':   # bert model
            print(f"start processing dataset with bert tokenizer...")
            start_time = time()
            self.tokenizer = BertTokenizerFast.from_pretrained(self.emb_type)
            # TODO: are values from 'token_type_ids' and 'attention_mask' helpful?
            # padding is done in dataloader
            self.data_idx = self.tokenizer(self.data_orig, max_length=maxlen, truncation=True)['input_ids']
            self.data_idx = [[s for s in sent_ids if s > 100] for sent_ids in self.data_idx]    # total len 8394039 -> 8392590
            self.data_length = [len(i) for i in self.data_idx]
            print(f"processing dataset with bert tokenizer takes {time() - start_time} seconds")
            

        else:
            raise Exception("choose an available pre-trained embedding: word2vec, glove, bert-base-uncased")
        
        # logging.info("sorting dataset ...")
        self.sort_dataset()
        '''
        for bert, after sorting, those comments with pure emojis are ordered in the first place
        '''

        # aspects
        # logging.info("processing aspects ...")
        self.aspects, flatten_seedwords = self.load_aspect_init(aspect_init_file)
        self.num_asp = len(self.aspects)
        self.vectorizer = CountVectorizer(vocabulary=sorted(list(set(flatten_seedwords))))
        self.vectorizer.fixed_vocabulary_ = True
        self.id2asp = {idx: feat for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.asp2id = {feat: idx for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.aspect_ids = [[self.asp2id[asp] for asp in aspect] for aspect in self.aspects]

        if self.emb_type.startswith('bert'):
            logging.info(f"[{domain}] dataset from following files: {data_file}, bert use own tokenizer")
        else:
            logging.info(f"[{domain}] dataset from following files: {data_file}, {vocab_file}, {supplement_file}, {aspect_init_file}")
        logging.info(f'[{domain}] dataset_size: {len(self.data_orig)}')
        logging.info(f'[{domain}] number of aspects: {len(self.aspect_ids)}')
        logging.info(f'[{domain}] number of unique seed words: {len(self.asp2id)}')


    def get_idx2asp(self):
        """idx2asp
        Returns:
            result: [bow_size], result[i] is the aspect of i-th seedword
        """
        result = []
        for feat in self.vectorizer.get_feature_names():
            for i in range(len(self.aspects)):
                if feat in self.aspects[i]:
                    result.append(i)
                    break
        return result

    def sort_dataset(self):
        '''sort data based on data_length'''
        sorted_idx = np.argsort(self.data_length)
        self.data_orig = [self.data_orig[i] for i in sorted_idx]
        self.data_idx = [self.data_idx[i] for i in sorted_idx]
        self.data_length = [self.data_length[i] for i in sorted_idx]

    @staticmethod
    def build_text_index(sentence, vocab):
        '''transform sentence string to index according to vocab'''
        # senc = tokenize_sentence(sentence)    # for oposum dataset don't use tokenizer
        senc = sentence.split()
        senc = lemmatize_sentence(senc)
        senc = remove_wordlist(senc, set(stopwords.words('english')))
        idx = [vocab.get(token, vocab['<UNK>']) for token in senc]  # not existing token is <UNK>
        if len(idx) == 0:
            idx = [vocab['<UNK>']]
        return idx

    @staticmethod
    def load_data(file):
        '''we use our own vocab, loading original maybe enough'''
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        orig = []
        for d in data['original']:
            if isinstance(d, list):
                orig.extend(d)
            elif isinstance(d, str):
                orig.append(d)
        # data = [s for d in data['original'] for s in d]
        return orig

    @staticmethod
    def load_aspect_init(file):
        with open(file) as f:
            text = f.read()
        text = text.strip().split('\n')
        result = [t.strip().split() for t in text]
        return result, [i for r in result for i in r]

    @staticmethod
    def build_shift_vocab_word2vec(model_file, num_tag=2):
        emb = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
        shift = num_tag
        vocab = {token: i + shift for i, token in enumerate(emb.wv.index2word)}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    @staticmethod
    def build_shift_vocab_glove(glove_filename, num_tag=2):
        shift = num_tag
        with open(glove_filename, 'rb') as f:
            glove = pickle.load(f)
        vocab = {token: i + shift for i, token in enumerate(glove.keys())}
        vocab['<PAD>'] = 0
        vocab['<UNK>'] = 1
        return vocab

    @staticmethod
    def load_vocab_from_file(file_name):
        vocab = {}
        with open(file_name, 'r') as f:
            for line in f:
                token, idx = line.strip().split('\t')
                vocab[token] = idx
        return vocab

    def __getitem__(self, index: int):
        bosw = self.vectorizer.transform([self.data_orig[index]]).toarray()[0]   # only 1 sentence in the list
        if self.emb_type == 'word2vec':
            idx = self.data_idx[index]
        elif self.emb_type == 'glove':
            idx = self.data_idx[index]
        else:
            '''
            TODO: what are those index smaller than 100 in BERT? -> emoji is converted to 100
            tokenizer.encode() return only 'input_ids'
            tokenizer(text) return dictionary with keys 'input_ids', 'token_type_ids' and 'attention_mask' 
            '''
            # idx = self.tokenizer.encode(
            #     self.data_orig[index], max_length=self.maxlen, padding=True, truncation=True)
            # idx = [i for i in idx if i > 100]
            idx = self.data_idx[index]
        # actual_len = len(idx) if len(idx) <= self.maxlen else self.maxlen
        actual_len = self.data_length[index]
        # idx = idx[:self.maxlen]
        # idx += [0] * (self.maxlen - len(idx))
        return (torch.from_numpy(bosw), torch.LongTensor(idx), torch.LongTensor([actual_len]))

    def __len__(self):
        return len(self.data_orig)


class TestDataset(Dataset):
    def __init__(self, data_base_dir, domain, stu_params, maxlen=128):
        # print('loading dataset...')
        # super(TestDataset, self).__init__(aspect_init_file, file, stu_params, maxlen)
        # self.data_orig, self.label = self.data_orig

        dataset_mode = 'test'
        if domain in oposum_domains:
            aspect_init_file = os.path.join(data_base_dir, f'{domain}.30.txt')
            data_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}.json')
        else:   # organic dataset
            aspect_init_file = os.path.join(data_base_dir, 'seedwords_30_coarse.txt')
            data_file = os.path.join(data_base_dir, f'annotated_{dataset_mode}_coarse.json')
        
        # self.maxlen = maxlen
        # data
        self.data_orig, self.label = self.load_data(data_file)

        # embedding
        self.emb_type = stu_params['pretrained']
        if self.emb_type == 'word2vec':
            # self.vocab = self.build_shift_vocab_word2vec(stu_params['wv_file'])
            # self.data_idx = [self.build_text_index(s, self.vocab) for s in self.data_orig]
            # self.data_length = [len(s) for s in self.data_idx]
            vocab_file = os.path.join(data_base_dir, f'{domain}_vocab_w2v.txt')
            self.vocab = self.load_vocab_from_file(vocab_file)
            if domain in oposum_domains:
                supplement_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}_supplement_w2v.pkl')
            elif domain == 'organic':
                supplement_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}_coarse_supplement_w2v.pkl')
            supplement = pickle_load(supplement_file)
            self.data_idx = supplement['data_idx']
            self.data_length = supplement['data_length']
        elif self.emb_type == 'glove':
            # self.vocab = self.build_shift_vocab_glove(stu_params['wv_file'])
            # self.data_idx = [self.build_text_index(s, self.vocab) for s in self.data_orig]
            # self.data_length = [len(s) for s in self.data_idx]
            vocab_file = os.path.join(data_base_dir, f'{domain}_vocab_glove.txt')
            self.vocab = self.load_vocab_from_file(vocab_file)
            supplement_file = os.path.join(data_base_dir, f'{domain}_{dataset_mode}_supplement_glove.pkl')
            supplement = pickle_load(supplement_file)
            self.data_idx = supplement['data_idx']
            self.data_length = supplement['data_length']
        else:   # bert model
            print(f"start processing dataset with bert tokenizer...")
            start_time = time()
            self.tokenizer = BertTokenizerFast.from_pretrained(self.emb_type)
            self.data_idx = self.tokenizer(self.data_orig, max_length=maxlen, truncation=True)['input_ids']
            self.data_idx = [[s for s in sent_ids if s > 100] for sent_ids in self.data_idx]
            self.data_length = [len(i) for i in self.data_idx]
            print(f"processing dataset with bert tokenizer takes {time() - start_time} seconds")
        
        self.sort_dataset()

        # aspects
        self.aspects, flatten_seedwords = self.load_aspect_init(aspect_init_file)
        self.num_asp = len(self.aspects)
        self.vectorizer = CountVectorizer(vocabulary=sorted(list(set(flatten_seedwords))))
        self.vectorizer.fixed_vocabulary_ = True
        self.id2asp = {idx: feat for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.asp2id = {feat: idx for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.aspect_ids = [[self.asp2id[asp] for asp in aspect] for aspect in self.aspects]

        logging.info(f"[{domain}] test dataset from following files: {data_file}, bert use own tokenizer")
        logging.info(f'[{domain}] test dataset size: {len(self.data_orig)}')
        # logging.info(f'[{domain}] number of aspects: {len(self.aspect_ids)}')
        # logging.info(f'[{domain}] number of unique seed words: {len(self.asp2id)}')

    def sort_dataset(self):
        sorted_idx = np.argsort(self.data_length)
        self.data_orig = [self.data_orig[i] for i in sorted_idx]
        self.data_idx = [self.data_idx[i] for i in sorted_idx]
        self.data_length = [self.data_length[i] for i in sorted_idx]
        self.label = [self.label[i] for i in sorted_idx]
    
    @staticmethod
    def load_data(file):
        with open(file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # deal with cases between a list of strings {'original': [str1, str2, ...]} or 
        # a list of lists of strings {'original': [[str1, str2, ...], [str1, str2, ...], ...]}
        orig = []
        label = []
        for d in data['original']:
            if isinstance(d, list):
                orig.extend(d)
            elif isinstance(d, str):
                orig.append(d)
        # label_shape = np.shape(data['label'])
        if isinstance(data['label'][0][0], int):
            label = data['label']
        elif isinstance(data['label'][0][0][0], int):
            label = [s for d in data['label'] for s in d]
        else:
            raise Exception("check data processing in test dataset")
        return orig, label

    def __getitem__(self, index: int):
        bosw = self.vectorizer.transform([self.data_orig[index]]).toarray()[0]
        if self.emb_type == 'word2vec':
            idx = self.data_idx[index]
        elif self.emb_type == 'glove':
            idx = self.data_idx[index]
        else:
            # idx = self.tokenizer.encode(
            #     str(self.data_orig[index]), max_length=self.maxlen, padding=True, truncation=True)
            # idx = [i for i in idx if i > 100]
            idx = self.data_idx[index]
        # actual_len = len(idx)
        actual_len = self.data_length[index]
        # idx = idx[:self.maxlen]
        # idx += [0] * (self.maxlen - len(idx))
        return (torch.LongTensor(idx), torch.LongTensor(bosw), torch.LongTensor(self.label[index]), torch.LongTensor([actual_len]))


if __name__ == '__main__':
    # bags: [31, 14, 16, 53, 300, 26, 22, 74, 106]
    # boots: [ 14, 106,  39,  49,  47, 303,  20,  68,  25]
    # tv: [ 28,  51,  31,  41,  95, 413,  31,  22, 101]
    from tqdm import tqdm
    torch.set_printoptions(profile="full")
    # ds = TestDataset('./data/seedwords/tv.5.txt', 'data/tv_test.json')

    # debug in root folder 
    # stu_param = {'pretrained': 'word2vec',
    #              'wv_file': './wv/oposum_w2v/bags_and_cases_tuned.bin',
    #              'pretrained_dim': 300,
    #              'num_aspect': 9,
    #              'freeze_emb': 1,
    #              'droppout': 0.5,
    #              'weight_decay': 0.1}
    stu_param = {'pretrained': 'bert-base-uncased',
                  }
    # ds = Dataset('./data/', 'bags_and_cases', stu_param)
    # ds = Dataset('./processed/', 'organic', stu_param)
    ds = Dataset('./processed', 'organic', stu_param)
    # test_ds = TestDataset('./LeverageJustAFewKeywords/data/', 'bags_and_cases', stu_param)
    cnt = 0
