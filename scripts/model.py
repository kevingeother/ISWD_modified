'''
code adapted from https://github.com/aqweteddy/LeverageJustAFewKeywords/blob/master/model.py
'''

from torch import nn
import torch.nn.functional as F
from transformers import AutoModel
import torch



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
        # assert result.shape == []
        mask = bow.sum(1) == 0  # [B]
        result[mask, self.general_asp] = 1  # pretend that general words appear once
        result = torch.softmax(result, -1)
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
        """
        zc = z * bow
        r = torch.sum((self.idx2asp == asp).float() * zc, -1)
        return r



class Student(nn.Module):
    def __init__(self, hparams) -> None:
        super(Student, self).__init__()
        self.hparams = hparams
        self.bert = AutoModel.from_pretrained(hparams['pretrained'])
        self.fc = nn.Sequential(
            nn.Linear(hparams['pretrained_dim'], hparams['num_aspect']))

    def forward(self, x):
        '''
        return:
            prob: [B, asp_cnt]
        '''
        x = F.dropout(self.bert(x)[0][:,0,:], 0.2)  # TODO: shape from BERT
        logits = self.fc(x)

        return torch.softmax(logits, dim=-1)


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
