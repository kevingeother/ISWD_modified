from train import Trainer
from utils import *
import os
from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
from tensorboardX import SummaryWriter
import numpy as np

'''
parameter search on organic dataset
'''

hparams = {
    'domain': 'organic',    # 'oposum', 'organic', 'bags_and_cases', etc
    'experiment_mode': 'one-time', # 'multi-times', 'one-time'
    'lr': 5e-4,
    'batch_size': 512,
    'inner_iter': 6,
    'epochs': 5,
    'gpu': '1',
    'student': {
        'pretrained': 'word2vec',
        'wv_path': '../wv/w2v_corpus_wotf1_wostw_tuned.bin',
        'wv_prefix': 'w2v_corpus_wotf1_wostw',
        'wv_mode': 'tuned',     # 'pretrained'
        'pretrained_dim': 300,
        'num_aspect': 9,
        'freeze_emb': 1,
        'dropout': 0.5,
        'weight_decay': 0.1,
    },
    'data_dir': '../processed/',
    'output_dir': '../experiments/param_search',
    'general_asp': 4,
}

aspect_name = read_aspect_name('../processed/organic_aspect_name_coarse.txt')
hparams['general_asp'] = aspect_name.index('general')
hparams['student']['num_aspect'] = len(aspect_name)

time_suffix = datetime.now().strftime('%y%m%d_%H%M%S')
base_dir = os.path.join(hparams['output_dir'], f"{time_suffix}-{hparams['student']['pretrained']}")

writer = SummaryWriter(base_dir)
print(f"logging dir: {base_dir}")
# TODO: run multiple times
for lr in [5e-4, 5e-3, 5e-2]:
    for dropout in [0.1, 0.2, 0.3, 0.4, 0.5]:
        for weight_decay in [0.01, 0.1, 1]:
            print(f"\nsetting: lr={lr}\tdropout={dropout}\tweight_decay={weight_decay}")
            param_suffix = f"lr{lr}-do{dropout}-wd{weight_decay}"
            hparams['output_dir'] = os.path.join(base_dir, param_suffix)
            hparams['lr'] = lr
            hparams['student']['dropout'] = dropout
            hparams['student']['weight_decay'] = weight_decay
            print(f"experiment setting: {hparams}\n")
            trainer = Trainer(hparams, 'cuda')
            result = trainer.train()
            writer.add_hparams(flatten_dict(hparams), {'micro-f1': np.max(result['micro_f1']), 
                                                       'agreement_ratio': np.max(result['agreement_ratio'])})

writer.close()
