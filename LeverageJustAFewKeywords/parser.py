import os, click
from train import Trainer
from config import hparams
import torch
from datetime import datetime

oposum_domains = ['bags_and_cases', 'bluetooth', 'boots', 'keyboards', 'tv', 'vacuums']
general_apsects = [4, 6, 5, 7, 5, 5]

@click.command()
@click.option('--domain', default=hparams['domain'])
@click.option('--experiment_mode', default=hparams['experiment_mode'])
@click.option('--lr', default=hparams['lr'])
@click.option('--batch_size', default=hparams['batch_size'])
@click.option('--inner_iter', default=hparams['inner_iter'])
@click.option('--epochs', default=3)
@click.option('--gpu', default='1', help='-1 is CPU')
@click.option('--pretrained', default=hparams['student']['pretrained'])
@click.option('--wv_path', default=hparams['student']['wv_path'])
@click.option('--wv_mode', default=hparams['student']['wv_mode'])
@click.option('--pretrained_dim', default=hparams['student']['pretrained_dim'])
@click.option('--num_aspect', default=hparams['student']['num_aspect'])
@click.option('--freeze_emb', default=hparams['student']['freeze_emb'])
@click.option('--dropout', default=hparams['student']['dropout'])
@click.option('--weight_decay', default=hparams['student']['weight_decay'])
@click.option('--data_dir', default=hparams['data_dir'])
@click.option('--output_dir', default=hparams['output_dir'])
@click.option('--general_asp', default=hparams['general_asp'])
def run_experiments(**kwargs):
    hparams = set_hparams(**kwargs)
    # # check arguments type
    # for k, v in hparams.items():
    #     print(k, '\t', type(v))
    #     if type(v) == dict:
    #         for a, b in v.items():
    #             print(a, '\t', type(b))
    # os.environ["CUDA_VISIBLE_DEVICES"] = gpu if gpu != -1 else None   # cause error
    click.echo(f'Now GPU: {torch.cuda.get_device_name(0)}')
    
    time_suffix = datetime.now().strftime('%y%m%d_%H%M%S')
    base_dir = os.path.join(hparams['output_dir'], f"{time_suffix}-{hparams['student']['pretrained']}")
    print(f"logging dir here: {base_dir}")
    # run for whole oposum dataset
    if hparams['domain'] == 'oposum':
        for i, domain in enumerate(oposum_domains):
            hparams['domain'] = domain
            hparams['general_asp'] = general_apsects[i]
            print(f"\nexperiments setting: {hparams}\n")
            if hparams['experiment_mode'] == 'multi-times':
                for i in range(5):
                    hparams['output_dir'] = os.path.join(base_dir, domain, f"v{i}")
                    trainer = Trainer(hparams, 'cuda' if hparams['gpu'] != '-1' else 'cpu')
                    trainer.train()
            else:
                hparams['output_dir'] = os.path.join(base_dir, domain)
                trainer = Trainer(hparams, 'cuda' if hparams['gpu'] != '-1' else 'cpu')
                trainer.train()

    # TODO: run for one domain in oposum
    elif hparams['domain'] in oposum_domains:
        print(f"experiments setting: {hparams}")
        hparams['output_dir'] = base_dir + '_' + hparams['domain']
        if hparams['experiment_mode'] == 'multi-times':
            temp_dir = hparams['output_dir']
            for i in range(5):
                hparams['output_dir'] = os.path.join(temp_dir, f"v{i}")
                trainer = Trainer(hparams, 'cuda')
                trainer.train()
        elif hparams['experiment_mode'] == 'once':
            trainer = Trainer(hparams, 'cuda' if hparams['gpu'] != '-1' else 'cpu')
            trainer.train()
        else:
            print("check experiment_mode variable...")

    # TODO: run for organic dataset
    elif hparams['domain'] == 'organic':
        print(f"experiments setting: {hparams}")
        hparams['output_dir'] = base_dir + '_' + hparams['domain']
        if hparams['experiment_mode'] == 'multi-times':
            temp_dir = hparams['output_dir']
            for i in range(5):
                hparams['output_dir'] = os.path.join(temp_dir, f"v{i}")
                trainer = Trainer(hparams, 'cuda')
                trainer.train()
        elif hparams['experiment_mode'] == 'once':
            trainer = Trainer(hparams, 'cuda' if hparams['gpu'] != '-1' else 'cpu')
            trainer.train()
        else: 
            print("check experiment_mode variable...")
    else:
        raise Exception("There is no such domain.")
    

def set_hparams(**kwargs):
    for key in kwargs.keys():
        if key in hparams.keys():
            hparams[key] = kwargs[key]
        if key in hparams['student'].keys():
            hparams['student'][key] = kwargs[key]
    return hparams


if __name__ == '__main__':
    run_experiments()