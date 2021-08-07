# Leveraging Just a Few Keywords for Fine-Grained Aspect Detection Through Weakly Supervised Co-Traing

---
**IMPORTANT**: Original code in this folder is from https://github.com/aqweteddy/LeverageJustAFewKeywords, thanks for the code. And we change the code according to our understanding of the paper and our actual requirements.

* [original paper](https://arxiv.org/pdf/1909.00415.pdf)
* I can't find the official implementation or any unofficial implementation.

## Dataset

* [OPOSUM](https://github.com/stangelid/oposum)

## requirements

* pytorch >= 1.5
* numpy
* h5py
* click

## How to Run

### preprocess data

* preprocess data follow [OPOSUM](https://github.com/stangelid/oposum)
    * hdf5 (train data and test data)
    * seed words
* extract preprocessed hdf5 data through through extract_data.py
```sh
python extract_data.py --source ./data/preprocessed/BOOTS_MATE.hdf5 --output ./data/boots_train.json
python extract_data.py --source ./data/preprocessed/BOOTS_MATE_TEST.hdf5 --output ./data/boots_test.json
```

### train

* you can set config in `config.py` or using arguments.
    * notice that the general aspect index is not same in every datasets.
* start training
    * `seed_words` is no weight.
```bash
python parser.py --train_file ./data/boots_train.json --test_file ./data/boots_test.json --save_dir ./ckpt/boots --aspect_init_file ./data/boots.30.txt --epochs 3
```
* `python parser.py --help` to see detail.

## Benchmark

### OPOSUM

* Bags: 0.59 (RandomSampler 50000 data run 3 epochs.)
* TV: 

## Some difference between the paper and this implementation

### RandomSampler

* random sample 50000 data every epochs.

### Teacher

* In paper sec.(3.1), 
> If no seed word appears in `s`, then the teacher predicts the "General" aspect by setting $q_i^k = 1$

but in this implementation,
> If no seed word appears in `s`, I let the teacher predicts like the seedword of general aspect appear one time.