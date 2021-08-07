import re
import torch
from dataset import NewsDataset

def calc_z(logits, bow):
        """z
        Args:Dataset
            logits: B, asp_cnt
            bow: B, len_bow
        Returns:
            : asp_cnt, len_bow
        """
        val, idx = logits.max(1)
        print(idx)
        # words_len = bow.shape[1]
        num_asp = logits.shape[1]
        # bsum = bow.sum(0).float()
        # bsum = bsum.masked_fill(bsum == 0., 1e-9)
        r = torch.stack([(bow[torch.where(idx == k)] > 0).float().sum(0) for k in range(num_asp)])
        print(r.shape, r)
        bsum = r.sum(-1).view(-1, 1)
        bsum = bsum.masked_fill(bsum == 0., 1e-9)
        return r / bsum

def z():
    logits = torch.softmax(torch.rand((8, 3)), -1) # 3 aspect
    bow2 = torch.randint(0, 2, (4, 5)) # 5 bow
    bow1 = torch.randint(0, 1, (4, 5)) # 5 bow
    bow = torch.cat([bow1, bow2], 0)
    print(bow)
    r = calc_z(logits, bow)
    print(r)


class Teacher(torch.nn.Module):
    def __init__(self, idx2asp, asp_cnt, general_asp):
        super(Teacher, self).__init__()
        self.idx2asp = idx2asp
        self.asp_cnt = asp_cnt
        self.general_asp = general_asp
    def forward(self, bow, z):
        """Teacher
        Args:
            bow (torch.tensor): [B, bow_size]
            zs  (torch.tensor): [num_asp, bow_size]
        Returns:
            : [B, asp_cnt]
        """
        q = torch.cat([self.cal_q(bow, z, k) for k in range(self.asp_cnt)], dim=1).to('cpu') # [B, asp_cnt]
        q[q.sum(1) == 0, self.general_asp] = 1e10 # set general aspect
        q = torch.softmax(q, dim=1)
        return q
    
    def cal_q(self, bow, z, k):
        """calc
        Args:
            bow (tensor): [B, bow_size]
            z (tensor): [bow_size]
            k (tensor): int
        """
        G = torch.where(self.idx2asp == k)[0]
        return (z[k, G] * bow[:, G]).sum(1).unsqueeze(1)


def teacher():
    from config import hparams
    from torch.utils.data import DataLoader
    ds = NewsDataset(hparams['aspect_init_file'], hparams['train_file'],
                          hparams['student']['pretrained'], hparams['maxlen'])
    dl = DataLoader(ds, batch_size=8)
    idx2asp = torch.tensor(ds.get_idx2asp())
    teacher = Teacher(idx2asp, 9, 5)
    z = torch.ones((9, idx2asp.shape[0]))
    for bow, ids in dl:
        print(bow)
        ans = teacher(bow, z)
        print(ans)
        break


import numpy as np
torch.set_printoptions(profile="full")

torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
teacher()