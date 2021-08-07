hparams = {
    # 'domain': 'oposum', # 'oposum', 'organic', 'bags_and_cases', etc
    'domain': 'bags_and_cases',
    'experiment_mode': 'multi-times', # 'multi-times', 'debug'
    'lr': 5e-5,
    'batch_size': 4,
    'inner_iter': 5,
    'epochs': 6,
    'gpu': '1',
    'student': {
        # 'pretrained': 'glove',   # word2vec, glove, bert-base-uncased,
        # 'wv_file': '../glove.6B/glove.6B.300d.txt',     # file for word embeddings
        'pretrained': 'word2vec',
        # 'wv_file': '../wv/w2v_corpus_wotf1_tuned.bin',
        'wv_path': '../wv/oposum_w2v',
        'wv_mode': 'tuned',     # 'pretrained'
        'pretrained_dim': 300,
        'num_aspect': 9,
        'freeze_emb': 1,
        'dropout': 0.5,
        'weight_decay': 0.1,
    },
    # 'description': 'bag_and_cases baseline',
    # 'save_dir': './ckpt/bags_and_cases',
    'data_dir': './data/',
    'output_dir': './experiments/',
    # 'aspect_init_file': './data/bags_and_cases.30.txt',
    # 'train_file': './data/bags_and_cases_train.json',
    # 'test_file': './data/bags_and_cases_test.json',
    'general_asp': 4,
    'maxlen': 40
}

# in local machine/gitlab repo, LeverageJustAFewKeywords is not the root folder