'''
code adapted from https://github.com/aqweteddy/LeverageJustAFewKeywords/blob/master/dataset.py
'''

import json

import numpy as np
import torch
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from torch.utils import data
from transformers import BertTokenizerFast
import random
from nltk.tokenize import sent_tokenize


class Dataset(data.Dataset):
    def __init__(self, aspect_init_file, file, pretrained='bert-base-uncased', maxlen=10):
        self.aspects, vocab = self.load_aspect_init(aspect_init_file)
        self.vectorizer = CountVectorizer(vocabulary=sorted(list(set(vocab))))
        self.maxlen = maxlen
        self.tokenizer = BertTokenizerFast.from_pretrained(pretrained)
        self.data = self.load_data(file)
        self.vectorizer.fixed_vocabulary_ = True
        self.id2asp = {idx: feat for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.asp2id = {feat: idx for idx, feat in enumerate(
            self.vectorizer.get_feature_names())}
        self.aspect_ids = [[self.asp2id[asp] for asp in aspect] for aspect in self.aspects]
        print(f'asp_category: {len(self.aspect_ids)}')
        print(f'bag_dim: {len(self.asp2id)}')


    def get_idx2asp(self):
        """idx2asp

        Returns:
            : bow_size
        """
        result = []
        for feat in self.vectorizer.get_feature_names():
            for i in range(len(self.aspects)):
                if feat in self.aspects[i]:
                    result.append(i)
                    break
        return result

    @staticmethod
    def load_data(file):
        if file.endswith(".json"):  # TODO: json file not compatible yet
            with open(file, 'r') as f:
                data = json.load(f)
            data = [s for d in data['original'] for s in d]
        elif file.endswith(".pkl"):
            df = pd.read_pickle(file)
            text = df.iloc[:, 0].astype('str').tolist()
            text.sort(key=len)
            data = []
            for s in text:
                s_list = sent_tokenize(s)
                data += s_list
            data.sort(key=len)
        return data

    @staticmethod
    def load_aspect_init(file):
        with open(file) as f:
            text = f.read()
        text = text.strip().split('\n')
        result = [t.strip().split() for t in text]
        return result, [i for r in result for i in r]

    def __getitem__(self, index: int):
        bosw = self.vectorizer.transform([self.data[index]]).toarray()[0]   # only 1 sentence in the list
        idx = self.tokenizer.encode(
            self.data[index], max_length=self.maxlen, padding=True, truncation=True)
        idx = [i for i in idx if i > 100]
        idx = idx[:self.maxlen]
        idx += [0] * (self.maxlen - len(idx))   # padding
        return torch.from_numpy(bosw), torch.LongTensor(idx)

    def __len__(self):
        return len(self.data)


class TestDataset(Dataset):
    def __init__(self, aspect_init_file, file, pretrained='bert-base-uncased', maxlen=20):
        print('loading dataset...')
        super(TestDataset, self).__init__(
            aspect_init_file, file, pretrained, maxlen)
        self.data, self.label = self.data
        print(len(self.data), len(self.label))

    @staticmethod
    def load_data(file):
        with open(file) as f:
            data = json.load(f)
        return [s for d in data['original'] for s in d], [s for d in data['label'] for s in d]

    def __getitem__(self, index: int):
        bosw = self.vectorizer.transform([self.data[index]]).toarray()[0]
        idx = self.tokenizer.encode(
            str(self.data[index]), max_length=self.maxlen, padding=True, truncation=True)
        idx = [i for i in idx if i > 100]
        idx = idx[:self.maxlen]
        idx += [0] * (self.maxlen - len(idx))
        return torch.LongTensor(idx), torch.LongTensor(bosw), torch.LongTensor(self.label[index])


if __name__ == '__main__':
    # bags: [31, 14, 16, 53, 300, 26, 22, 74, 106]
    # boots: [ 14, 106,  39,  49,  47, 303,  20,  68,  25]
    # tv: [ 28,  51,  31,  41,  95, 413,  31,  22, 101]
    from tqdm import tqdm
    torch.set_printoptions(profile="full")
    # ds = TestDataset('./data/seedwords/tv.5.txt', 'data/tv_test.json')
    cnt = 0
    ds = Dataset('../processed/seedwords_5.txt', '../processed/processed_comment.pkl')
