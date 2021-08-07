from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
import torch
import gensim
import csv
import numpy as np
from collections import Counter
from nltk.corpus import brown
from mittens import GloVe, Mittens
# from sklearn.feature_extraction import stop_words
from sklearn.feature_extraction.text import CountVectorizer
import logging
import os
import pickle

logging.basicConfig(level=logging.INFO)

class Teacher(torch.nn.Module):
    def __init__(self, idx2asp, asp_cnt, general_asp, device) -> None:
        super(Teacher, self).__init__()
        self.idx2asp = idx2asp
        self.asp_cnt = asp_cnt
        self.general_asp = general_asp
        self.device = device
    
    def forward(self, bow, zs):
        """Teacher
        Args:
            bow (torch.tensor): [B, bow_size]
            zs  (torch.tensor): [num_asp, bow_size]
        Returns:
            result: [B, asp_cnt]
        """
        # for each aspect
        result = torch.stack([self.calc(bow, zs[i,:], i) for i in range(self.asp_cnt)], dim=-1) # [B, asp_cnt]
        # print(result.shape)
        mask = bow.sum(1) == 0  # bow: [B, bow_size] -> mask: [B]
        # result[mask, self.general_asp] = 1 # pretend that general words appear once
        result = torch.softmax(result, -1)
        result[mask, :] = 0
        result[mask, self.general_asp] = 1
        # result[mask, self.general_asp] = 1
        # result[mask, self.general_asp+1:] = 0
        # result[mask, :self.general_asp] = 0

        return result
    
    def calc(self, bow, z, asp):
        """calc for each aspect
        Args:
            bow (tensor): B, bow_size
            z (tensor): bow_size
            asp (tensor): int
            self.idx2asp: bow_size
        return:
            r: [B]
        math expression: nominator of eq. 1
            $$
            \sum_{j=1}^D \mathbb{1}\{ j \in G_k \} \cdot c_i^j
            $$
        """
        zc = z * bow
        r = torch.sum((self.idx2asp == asp).float() * zc, -1)
        return r


class Student(nn.Module):
    def __init__(self, hparams, domain) -> None:
        super(Student, self).__init__()
        self.emb_type = hparams['pretrained']
        if os.path.isdir(hparams['wv_path']):
            self.wv_file = os.path.join(hparams['wv_path'], f"{domain}_{hparams['wv_mode']}.bin")
        elif os.path.isfile(hparams['wv_path']):
            self.wv_file = hparams['wv_path']
        self.dropout = hparams['dropout']
        num_general_tag = 2     # <PAD>, <UNK>

        if self.emb_type == 'word2vec':
            try:
                w2v = gensim.models.KeyedVectors.load_word2vec_format(self.wv_file, binary=True)
            except:
                raise Exception(f"check your word vector file {self.wv_file}")
            vocab_size = len(w2v.wv.vectors)
            emb_dim = w2v.wv.vectors.shape[1]
            self.encoder = nn.Embedding(vocab_size + num_general_tag, emb_dim, padding_idx=0)
            self.encoder.weight.data[2:].copy_(torch.from_numpy(w2v.wv.vectors))
            self.encoder.weight.data[0] = torch.zeros(emb_dim)
            self.encoder.weight.data[1] = torch.rand(emb_dim) - 0.5     # TODO: emb for <UNK>
            logging.info(f"[{domain}] load pre-defined word vectors from {self.wv_file}")
        elif self.emb_type == 'glove':
            glove = self.load_glove(self.wv_file)
            vocab_size = len(glove)
            glove_vectors = np.array(list(glove.values()))
            emb_dim = glove_vectors.shape[1]
            self.encoder = nn.Embedding(vocab_size + num_general_tag, emb_dim, padding_idx=0)
            self.encoder.weight.data[2:].copy_(torch.from_numpy(glove_vectors))
            self.encoder.weight.data[0] = torch.zeros(emb_dim)
            self.encoder.weight.data[1] = torch.rand(emb_dim) - 0.5  # TODO: emb for <UNK>
        else:   # bert models
            self.encoder = AutoModel.from_pretrained(hparams['pretrained'])
            logging.info(f"[{domain}] load pre-defined bert model")
        
        # options for freezing embeddings
        self.freeze_emb = int(hparams['freeze_emb'])
        # print(type(self.freeze_emb))
        if self.freeze_emb:
            for param in self.encoder.parameters():
                param.requires_grad = False
        # self.print_grad_weight()
        # fully connected classifier
        self.fc = nn.Sequential(
            nn.Linear(hparams['pretrained_dim'], hparams['num_aspect']))
        # initialization of fc layer
        for m in self.fc:
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x, x_len):
        """
        return:
            prob: [B, asp_cnt]
        variable:
            x: index of tokens [B, seq_length]
            x_len: actual length of sentences [B]
            self.encoder(x)[0]: last_hidden_state, [B, sequence_length, hidden_size]
            [:, 0, :]: [CLS] tag
        """
        if self.emb_type == 'word2vec':
            x = self.encoder(x)     # [B, seq_length, emb_dim]
            x = x.sum(dim=1) / x_len.reshape(-1, 1)   # [B, emb_dim]  # average over actual length
        elif self.emb_type == 'glove':
            x = self.encoder(x)  # [B, seq_length, emb_dim]
            x = x.sum(dim=1) / x_len.reshape(-1, 1)
        else:
            x = self.encoder(x)[0][:, 0, :]
        x = F.dropout(x, self.dropout)
        logits = self.fc(x)
        return torch.softmax(logits, dim=-1)

    def print_grad_weight(self):
        for n, param in self.encoder.named_parameters():
            if param.requires_grad:
                print(n)

    @staticmethod
    def load_glove(glove_filename):
        with open(glove_filename, 'rb') as f:
            return pickle.load(f)


if __name__ == '__main__':
    device = 'cpu'

    idx2asp = torch.randint(0, 5, (10,))  # 10 seed word, 5 aspect
    print(idx2asp)
    bow1 = torch.randint(0, 1, (4, 10))  # bow_size=10
    bow2 = torch.randint(0, 3, (4, 10))  # bow_size=10
    bow = torch.cat([bow1, bow2], 0)
    print(f'bow: {bow}')
    z = torch.softmax(torch.ones((5, 10)), 0)
    teacher = Teacher(idx2asp, 5, 1, device)
    r = teacher(bow, z)
    print(r)
    # print(r.sum(-1))
