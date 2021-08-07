**Weakly Supervised Aspect Extraction Using a Student-Teacher Co-Training Approach**

--------

Overview tasks:
- derive seed words using word vectors, NMF and clarity scoring function
- text pre-processing and word embedding fine-tunning on OPOSUM and organic dataset
- develop student-teacher co-training approach and verify correctness with OPOSUM
- apply the approach to organic dataset with three different word embedding models, i.e. word2vec, glove and BERT
# Instructions 

## Running Experiments

```bash
conda create -n nlp python=3.7
conda activate nlp
pip install -r requirements.txt
```

Run `parser.py` with appropriate arguments.

Ex:

```bash
python parser.py --domain organic --experiment_mode once --freeze_emb 1 --lr 5e-4 --weight_decay 0.01 --dropout 0.4 \
    --batch_size 8 --inner_iter 1 --epochs 4 --pretrained glove --wv_path ../wv/glove_corpus_wotf1_wostw_pretrained.bin \
    --data_dir ../processed --general_asp 0 --num_aspect 6
```

- --domain 		-> organic, oposum, bags_and_cases etc
- --experiment_mode 	-> multi-times, once -> run the set of experiments once or multiple independent times
- --freeze_emb		-> 1 to freeze embeddings. 0 otherwise.
- --pretrained		-> glove, word2vec, bert-base-uncased
- --wv_path		-> #path/file for word embeddings e.g. '../wv/w2v_corpus_wotf1_tuned.bin', or  '../wv/glove_corpus_wotf1_wostw_pretrained.bin'
- --epochs		-> number of epochs
- --inner_iter		-> number of inner iterations
- --data_dir		-> path for pre-processed data
- --general_asp		-> index of general aspect of dataset -> 0 for organic
- --num_aspect		-> number of aspects of interest -> 6 for organic

Sample runs with setup environments are available in `scratch_experiments.ipynb` Jupyter Notebook.

## Jupyter Notebooks
Below is the description of available jupyter notebooks
- `Clarity Scoring.ipynb`: derive seed words with clarity scoring function 
- `Clarity_Scoring_coarse.ipynb`: derive seed words with clarity scoring function using coarse attributes from annotated dataset
- `draft_plot.ipynb`: some functions for plotting errorbar
- `extract_from_json.ipynb`: extract useful content from organic dataset, clean up articles, output json files for follow tasks
- `extractAnnotatedData.ipynb`: create organic test set
- `learn_transformers.ipynb`: experience following the official tutorial
- `NMF.ipynb`: derive seed words using NMF
- `oposum_build_corpus.ipynb`: prepare data for fine-tuning word embeddings
- `oposum_glove_finetune.ipynb`: fine-tune glove in oposum dataset
- `oposum_w2v_fine-tune.ipynb`: fine-tune word2vec in oposum datset
- `organic_create_json.ipynb`: create organic training set, sentencize and remove sentences in test set
- `organic_experiments.ipynb`: run experiments on the whole approach and teacher evaluation in organic dataset
- `process_oposum_dataset.ipynb`: transform oposum dataset into integer index, save useful files for fast loading
- `process_organic_dataset.ipynb`: the same as above but apply to organic dataset
- `WordEmbeddings.ipynb`: text pre-processing and fine-tune word2vec on organic dataset
- `WV_cluster.ipynb`: apply k-means to fine-tuned word embeddings

- `draft_UnofficialLeverage.ipynb`: code debugging, data inspection, etc.
- `change_trials`: experiments on different ideas/variants to change the unofficial code

# Folder Structure

```bash
.
├─annotated-dataset
│  ├─annotated_3rd_round
│  ├─...
│  └─...
├─curated-source-dataset
│  ├─english
│  └─german
├─glove.6B                      # glove pre-trained model
├─ISWD	                        # code from official repo, for faster accessibility
├─JupyterNotebook               # jupyter notebook scripts for various tasks, each with description inside
├─LeverageJustAFewKeywords      # code from unofficial repo, changed for our usage
├─oposum_scripts                # pre-processing code from summarizing opinions paper
├─output                        # intermediate results and other output files
├─presentation                  # presentation
├─processed                     # processed data and other useful files
├─scripts                       # scripts intended 
└─wv                            # word vectors for oposum and organic dataset
```

**Note**: 
- code in `./LeverageJustAFewKeywords` is originated from https://github.com/aqweteddy/LeverageJustAFewKeywords. Further information can be checked in `README.md` in the subfolder.
- folder `./ISWD` is from official code https://github.com/gkaramanolakis/ISWD, added with comments
