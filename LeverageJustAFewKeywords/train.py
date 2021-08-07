import json
import logging
import time
from collections import Counter
from torch.nn.utils.rnn import pad_sequence
import torch
import os
import numpy as np

from torch import nn, optim
from torch.utils import data
# from torch.utils.data import sampler
from tqdm import tqdm

from dataset import TestDataset, Dataset
from model import Student, Teacher
from utils import *

from sklearn.metrics import f1_score, recall_score, precision_score, confusion_matrix

# from tensorboardX import SummaryWriter
from torch.utils.tensorboard import SummaryWriter

os.environ["TOKENIZERS_PARALLELISM"] = "false"
logging.basicConfig(level=logging.INFO)
torch.set_printoptions(profile="full")
logging.getLogger('transformers').setLevel(logging.WARNING)

class EntropyLoss(nn.Module):
    '''
    official code use SmoothCrossEntropy
    '''
    def __init__(self):
        super(EntropyLoss, self).__init__()

    def forward(self, p, q):
        '''
        variable:
            q: target
            p: input
        '''
        b = q * torch.log(p)
        b = -1. * b.sum()
        b /= len(p)

        return b


class Trainer:
    def __init__(self, hparams, device, version=None) -> None:
        self.hparams = hparams
        # self.hparams['version'] = version
        self.domain = hparams['domain']
        self.inner_iter = hparams['inner_iter']
        self.epochs = hparams['epochs']
        self.output_dir = hparams['output_dir']
        self.device = device
        self.asp_cnt = self.hparams['student']['num_aspect']
        self.start = time.time()
        logging.info('loading dataset...')
        self.ds = Dataset(hparams['data_dir'], hparams['domain'], hparams['student'])
        self.test_ds = TestDataset(hparams['data_dir'], hparams['domain'], hparams['student'])
        self.test_loader = data.DataLoader(self.test_ds, batch_size=500, num_workers=2,
                                            collate_fn=self.test_collate, drop_last=False)  # colab warning for 2 workers

        # logging.info(f'dataset_size: {len(self.ds)}')

        logging.info('loading model...')
        self.idx2asp = torch.tensor(self.ds.get_idx2asp()).to(self.device)
        self.teacher = Teacher(self.idx2asp, self.asp_cnt,
                               self.hparams['general_asp'], self.device).to(self.device)
        self.student = Student(hparams['student'], self.domain).to(self.device)
        # self.student_opt = optim.Adam(self.student.parameters(), hparams['lr'])
        self.student_opt = optim.Adam(filter(lambda p:p.requires_grad, self.student.parameters()), 
                                        lr=hparams['lr'], weight_decay=hparams['student']['weight_decay'])
        self.criterion = EntropyLoss().to(self.device)
        # self.criterion = nn.BCELoss(reduction='sum').to(self.device)

        self.writer = SummaryWriter(log_dir=self.output_dir)

        self.z = self.reset_z()
        logging.debug(f'__init__: {time.time() - self.start}')

    def save_config(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        with open(os.path.join(path, 'config.json') ,'w') as f:
            json.dump(self.hparams, f)
    
    def save_model(self, path):
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        # with open(os.path.join(path, 'config.json'), 'w') as f:
        #     json.dump(self.hparams, f)
        torch.save({'teacher_z': self.z, 'student': self.student.state_dict()}, os.path.join(
            path, 'teacher_student.pt'))

    def train_loader(self, ds):
        # sampler = data.RandomSampler(ds, replacement=True, num_samples=10000)
        return data.DataLoader(ds, batch_size=self.hparams['batch_size'], num_workers=2,
                                collate_fn=self.train_collate, drop_last=False)    # colab warning for 2 workers, shuffle=False
    
    def train(self, epochs=None):
        if not epochs:
            epochs = self.epochs
        prev_best = torch.tensor([-1])
        loader = self.train_loader(self.ds)
        print(f"initial micro f1 score from teacher: {self.teacher_test()}")
        loss_list = []
        agr_list = []
        micro_f1_list = []
        micro_f1_teacher_list = []
        macro_f1_list = []
        precision_list = []
        recall_list = []
        pred_dist = []
        conf_mat_list = []
        for epoch in range(epochs):
            logging.info(f'ISWD iteration: {epoch}')
            # training
            loss_ep, agr_ratio = self.train_per_epoch(loader)
            # loss_ep, agr_ratio = self.train_per_ISWD(loader)
            print(f"z: {self.z.sum(-1)}")
            # testing
            result_dict = self.test()
            # result handling
            loss_list.append(loss_ep)
            agr_list.append(agr_ratio.item())
            micro_f1_list.append(result_dict['micro_f1'])
            micro_f1_teacher_list.append(result_dict['micro_f1_teacher'])
            macro_f1_list.append(result_dict['every_f1'].mean())
            precision_list.append(result_dict['precision'])
            recall_list.append(result_dict['recall'])
            pred_dist.append(result_dict['prediction_distribution'])
            conf_mat_list.append(result_dict['confusion_matrix'])
            # logging.info(f'epoch: {epoch}, f1_mid: {result_dict["micro_f1"]:.3f}, prev_best: {prev_best.item():.3f}')
            if prev_best < result_dict['micro_f1']:
                self.save_model(self.output_dir)
                prev_best = result_dict['micro_f1']
        for i in range(epochs):
            logging.info(f"epoch {i}:\tloss: {np.mean(loss_list[i]):.3f}\tscore: {micro_f1_list[i]:.3f}\tagreement_ratio: {agr_list[i]:.3f}")
        # print(loss_list, '\n', micro_f1_list, '\n', macro_f1_list, '\n', precision_list, '\n', recall_list, '\n', agr_list)
        self.train_result = {'loss': loss_list,
                            'agreement_ratio': agr_list,
                            'micro_f1': micro_f1_list,
                            'macro_f1': macro_f1_list,
                            'precision': precision_list,
                            'recall': recall_list,
                            'ground_class_distribution': result_dict['ground_class_distribution'],
                            'prediction_distribution': pred_dist,
                            'confusion matrix': conf_mat_list}
        self.save_config(self.output_dir)
        pickle_save(self.train_result, os.path.join(self.output_dir, 'result.pkl'))
        self.result_writer()
        self.writer.close()
        return self.train_result

    def train_per_ISWD(self, loader):
        # pbar = tqdm(total=len(loader))
        loss_out_loop = []
        for i in range(self.inner_iter):
            logging.info(f"inner epoch: {i}")
            loss_ep = []
            # train student
            train_bar = tqdm(total=len(loader))
            for x_bow, x_id, x_len in loader:
                x_bow, x_id, x_len = x_bow.to(self.device), x_id.to(self.device), x_len.to(self.device)
                self.student_opt.zero_grad()
                t_logits = self.teacher(x_bow, self.z)
                s_logits = self.student(x_id, x_len)
                loss = self.criterion(s_logits, t_logits)
                loss_ep.append(loss.item())
                loss.backward()
                self.student_opt.step()
                train_bar.update(1)
                train_bar.set_description(f'loss: {loss:.3f}')
            train_bar.set_description(f'avg loss: {np.mean(loss_ep):.3f}')
            train_bar.close()
            aver_loss = np.mean(loss_ep)
            loss_out_loop.append(aver_loss)
            # student converge
            if i > 0 and abs(loss_out_loop[i-1] - aver_loss) / aver_loss < 0.001:
                break
        print(loss_out_loop)
        # inference with student
        s_logits_ep = []
        x_bow_ep = []
        for x_bow, x_id, x_len in tqdm(loader):
            x_bow, x_id, x_len = x_bow.to(self.device), x_id.to(self.device), x_len.to(self.device)
            with torch.no_grad():
                s_logits = self.student(x_id, x_len)
            s_logits_ep.append(s_logits)
            x_bow_ep.append(x_bow)
        # calculate z
        s_logits_ep = torch.cat(s_logits_ep, dim=0)
        x_bow_ep = torch.cat(x_bow_ep, dim=0)
        t_logits_ep_old = self.teacher(x_bow_ep, self.z)
        self.z = self.calc_z(s_logits_ep, x_bow_ep)
        # update teacher
        t_logits_ep_new = self.teacher(x_bow_ep, self.z)
        # TODO use old or new teacher to compare teacher and student -> use old one
        agreement_ratio_old = (s_logits_ep.max(-1)[1] == t_logits_ep_old.max(-1)[1]).sum() / len(s_logits_ep)
        agreement_ratio_new = (s_logits_ep.max(-1)[1] == t_logits_ep_new.max(-1)[1]).sum() / len(s_logits_ep)
        print(f"{agreement_ratio_old*100:.2f}% ({agreement_ratio_new*100:.2f}% after updating z) of samples have same result from student and teacher.")
        return loss_out_loop, agreement_ratio_old

    def train_per_epoch(self, loader):
        losses = []
        agr_ratio_list = []
        pbar = tqdm(total=len(loader))

        for x_bow, x_id, x_len in loader:
            x_bow, x_id, x_len = x_bow.to(self.device), x_id.to(self.device), x_len.to(self.device)
            loss, agr_ratio = self.train_step(x_bow, x_id, x_len)
            losses.append(loss.item())
            agr_ratio_list.append(agr_ratio)
            pbar.update(1)
            pbar.set_description(f'loss:{loss.item():.3f}')
        pbar.close()
        losses = sum(losses) / len(losses)
        avg_agr_ratio = sum(agr_ratio_list) / len(agr_ratio_list)
        logging.info(f'train_loss: {losses}; agreement_ratio: {avg_agr_ratio}')
        return losses, avg_agr_ratio

    def reset_z(self):
        z = torch.ones(
            (self.asp_cnt, len(self.ds.asp2id))).to(self.device)
        return torch.softmax(z, dim=-1)
        # return z
    
    def train_step(self, x_bow, x_id, x_len):
        '''
        for each batch
        Issues
            1. apply the Iterative Seed Word Distillation to each batch VS to each epoch ?
            2. when to reset z
        '''
        # apply teacher
        self.z = self.reset_z()
        t_logits = self.teacher(x_bow, self.z)  # [B, asp_cnt]
        loss = 0.
        prev = -1
        # print()
        for i in range(self.inner_iter):
            # train student Eq. 2
            self.student_opt.zero_grad()
            s_logits = self.student(x_id, x_len)   # [B, asp_cnt]
            loss = self.criterion(s_logits, t_logits)
            # print(f'bow: {x_bow}')
            # print(f'z: {self.z}')
            # print(f'teacher:{t_logits.max(-1)[1]}')
            # print(f'x_id{x_id}')
            # print(f'student:{s_logits.max(-1)[1]}')
            loss.backward()
            self.student_opt.step()
            tmp = (t_logits.max(-1)[1] == s_logits.max(-1)[1]).sum()    # number of coincide
            if tmp == prev or tmp.item() == t_logits.shape[0]:
                break
            prev = tmp
            # update teacher Eq.4
            self.z = self.calc_z(s_logits, x_bow)

            # apply teacher Eq. 3
            t_logits = self.teacher(x_bow, self.z)
        return loss, tmp / len(t_logits)

    def test(self):
        result = []
        result_teacher = []
        ground = []
        self.student.eval()
        for batch in self.test_loader:
            idx, bow, labels, act_len = batch
            idx = idx.to(self.device)
            bow = bow.to(self.device)
            act_len = act_len.to(self.device)
            with torch.no_grad():
                logits = self.student(idx, act_len) # [B, n_asp]
                logits_teacher = self.teacher(bow, self.z)
            result.append(logits.max(-1)[1].detach().cpu()) # [B]
            result_teacher.append(logits_teacher.max(-1)[1].detach().cpu())
            ground.append(labels.max(-1)[1])    # [B]

        result = torch.cat(result, dim=0)
        result_teacher = torch.cat(result_teacher, dim=0)
        ground = torch.cat(ground, dim=0)
        micro_f1 = f1_score(ground.numpy(), result.numpy(), average='micro')
        micro_f1_teacher = f1_score(ground.numpy(), result_teacher.numpy(), average='micro')
        every_f1 = f1_score(ground.numpy(), result.numpy(), average=None)
        recall = recall_score(ground.numpy(), result.numpy(), average=None)
        precision = precision_score(ground.numpy(), result.numpy(), average=None)
        pred_dist = Counter(result.numpy())     # distribution
        pred_teacher_dist = Counter(result_teacher.numpy())
        ground_dist = Counter(ground.numpy())
        conf_mat = confusion_matrix(y_true=ground.numpy(), y_pred=result.numpy())
        print(f'testing result: micro f1 score from teacher: {micro_f1_teacher}\tmicro f1 score from student: {micro_f1}')
        eval_dict = {'micro_f1': micro_f1,
                     'micro_f1_teacher': micro_f1_teacher,
                     'every_f1': every_f1, 
                     'recall': recall,  # a list for each aspect
                     'precision': precision,    # a list for each aspect
                     'ground_class_distribution': dict(ground_dist),
                     'prediction_distribution': dict(pred_dist),
                     'teacher_prediction_distribution': dict(pred_teacher_dist),
                     'confusion_matrix': conf_mat}
        self.student.train()
        return eval_dict

    def teacher_test(self):
        result_teacher = []
        ground = []
        for batch in self.test_loader:
            idx, bow, labels, act_len = batch
            bow = bow.to(self.device)
            logits_teacher = self.teacher(bow, self.z)
            result_teacher.append(logits_teacher.max(-1)[1].detach().cpu())
            ground.append(labels.max(-1)[1])    # [B]
        result_teacher = torch.cat(result_teacher, dim=0)
        ground = torch.cat(ground, dim=0)
        micro_f1_teacher = f1_score(ground.numpy(), result_teacher.numpy(), average='micro')
        return micro_f1_teacher

    def calc_z(self, logits, bow):
        """z
        Args:
            logits: B, asp_cnt
            bow: B, bow_size
        Returns:
            z: asp_cnt, bow_size
        """
        val, idx = logits.max(1)    # [B]
        num_asp = logits.shape[1]
        r = torch.stack([(bow[torch.where(idx == k)] > 0).float().sum(0)    # [bow_size]
                         for k in range(num_asp)])  # [asp_cnt, bow_size], default dim=0
        # change the way of summation, unofficial repo wrong, because after changing performance gets better
        bsum = r.sum(-1).view(-1, 1)    # [asp_cnt, 1]
        # bsum = r.sum(0).view(1, -1)     # [1, bow_size]
        bsum = bsum.masked_fill(bsum == 0., 1e-10)
        z = r / bsum
        # z = torch.softmax(r, -1)
        # print(f'z: {z}')
        return z

    def result_writer(self):
        # SummaryWriter
        n_iter = self.epochs
        r = self.train_result
        for i in range(n_iter):
            self.writer.add_scalar('micro_f1', r['micro_f1'][i], i)
            self.writer.add_scalar('macro_f1', r['macro_f1'][i], i)
            self.writer.add_scalar('agreement_ratio', r['agreement_ratio'][i], i)
            self.writer.add_scalars('precision', list2dict(r['precision'][i]), i)
            self.writer.add_scalars('recall', list2dict(r['recall'][i]), i)
            for j, loss in enumerate(r['loss'][i]):
                self.writer.add_scalars('loss', {f"{i}-th iter": loss}, j)

    @staticmethod
    def train_collate(batch):
        '''(bosw, idx, actual_len)'''
        assert isinstance(batch, list)
        bosw_batch = torch.stack([b[0] for b in batch])
        idx_batch = pad_sequence([b[1] for b in batch], batch_first=True)
        actual_len_batch = torch.tensor([b[2] for b in batch], dtype=torch.int64)
        assert idx_batch.shape[0] == len(bosw_batch)
        return bosw_batch, idx_batch, actual_len_batch

    @staticmethod
    def test_collate(batch):
        '''(idx, bows, label, actual_len)'''
        assert isinstance(batch, list)
        idx_batch = pad_sequence([b[0] for b in batch], batch_first=True)
        bosw_batch = torch.stack([b[1] for b in batch])
        label_batch = torch.stack([b[2] for b in batch])
        actual_len_batch = torch.tensor([b[3] for b in batch], dtype=torch.int64)
        assert idx_batch.shape[0] == len(bosw_batch)
        assert idx_batch.shape[0] == len(label_batch)
        return idx_batch, bosw_batch, label_batch, actual_len_batch

    def run_multi_times(self, times=5):
        result_list = []
        for i in range(times):
            result = self.train()
        result_list.append(result)
        return result_list


if __name__ == '__main__':
    # debug in root folder
    hparams = {
        'domain': 'organic', #'bags_and_cases',
        'experiment_mode': 'once', # 'multi-times', 'once'
        'lr': 5e-5,
        'batch_size': 4,
        'inner_iter': 5,
        'epochs': 6,
        'gpu': '1',
        'student': {
            'pretrained': 'bert-base-uncased',
            'wv_path': '../wv/w2v_corpus_wotf1_tuned.bin',
            'wv_mode': 'tuned',     # 'pretrained'
            'pretrained_dim': 768,
            'num_aspect': 6,
            'freeze_emb': 1,
            'dropout': 0.5,
            'weight_decay': 0.01,
        },
        'data_dir': './processed/',
        'output_dir': './experiments/',
        'general_asp': 0,
        # 'maxlen': 40
    }
    trainer = Trainer(hparams, 'cuda')
    trainer.train()