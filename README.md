# AIM

This repo includes codes for reproducing the experiments in the paper Additive Instance-wise Approach to Multi-class Model Interpretation.

## Dependencies
AIM requires Python 3.7+ and the following packages

- `spacy`
- `nltk`
- `lemminflect`
- `numpy`
- `pandas`
- `tqdm`
- `torch`
- `torchtext`

Download this to use `spacy` tokenizer
```
python -m spacy download en_core_web_sm
```

Replicating the experiments on baselines in this repo further requires

- `lime`
- `tensorflow==1.15.0`
- `keras==2.0.0`

Or you can run the following command
```
git clone https://github.com/isVy08/AIM
cd AIM
pip install -r requirements.txt
```

**The following sections focus on experiments for texts. For details on experiments for MNIST dataset, please refer to `mnist/`.**


## Data
`data_generator.py` provides scripts for downloading datasets and training a tokenizer. 
IMDB and AG News are available in `torchtext` library while HateXplain can be obtained from [HateXplain repo](https://github.com/hate-alert/HateXplain/tree/master/Data) (Mathew et al. 2021). 

For pre-processed datasets and a pre-trained tokenizer used in our experiment, refer to this [Google Drive collection](https://drive.google.com/drive/folders/19IkiCWE-8g-XB0Mu4GAw2RgoejK3g6yg?usp=sharing).
<br>Download the datasets to `data/` and the pre-trained tokenizer to `model/`.

## Model
Configurations for black-box models and model explainers are given in `config/`. 

### Black-box Models
Pre-trained black-box models for each dataset are available in their respective folders. 
If you need to train them from sratch, please check `blackbox.py` and `train_blackbox.py`.

For example, to train a bidirectional GRU on IMDB dataset, run the following command

```
python train_blackbox.py config/WordGRU.json train
```

Before training model explainers, you need to obtain the black-box's predictions. Do run
```
python train_blackbox.py config/WordGRU.json val
```
and predictions will be generated in the same format as the original dataset under the name `WordGRU.pickle`. Again, you can directly download the predictions from [Google Drive](https://drive.google.com/drive/folders/19IkiCWE-8g-XB0Mu4GAw2RgoejK3g6yg?usp=sharing) and place them inside the corresponding folder, i.e., `data/imdb/`.

### Model Explainers
Our architecture is described in `explainer.py`. To train a model explainer for a dataset e.g., IMDB, do
```
python main.py config/imdb.json
```
You will find the trained models inside their respective directory `model/`


## Inference
`infer.py` provides instructions on how to perform inference for **AIM** and other baselines. For example on IMDB test set, run

```
python infer.py imdb
```
and feature weights will be written in a text file `data/imdb/aim.txt`. 


# Evaluation
First, we need a list of stopwords and [WordNet](https://wordnet.princeton.edu/) database. The [Google Drive folder](https://drive.google.com/drive/folders/1LfEAcN1DU9PKYRiY0e8dZ8zTcG0k1Mfv?usp=sharing) provides a curated list of stopwords and a shortcut `dict` object to Wordnet database. Download and place them inside `model/`.

To evaluate **AIM** on IMDB test set, run 

```
python evaluate.py imdb data/imdb/aim.txt
``` 

On how to run experiments on the baseline models, please refer to `baseline/`. 
<br>The codes are gratefully adapted from [L2X repo](https://github.com/Jianbo-Lab/L2X), [LIME repo](https://github.com/marcotcr/lime-experiments) and [VIBI repo](https://github.com/SeojinBang/VIBI).


# Citation
If you use the codes or datasets in this repository, please cite our paper.
