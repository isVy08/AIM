# AIM

This repo includes codes for reproducing the experiments in the paper [Additive Instance-wise Approach to Multi-class Model Interpretation](https://github.com/isVy08/AIM/edit/master/README.md).

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


Or you can run the following command
```
git clone https://github.com/isVy08/AIM
cd AIM
pip install -r requirements.txt
```
## Data
The script `data_generator.py` provides scripts for downloading datasets and training a tokenizer. 
IMDB and AG News are available in `torchtext` library while HateXplain is taken from [HateXplain repo](https://github.com/hate-alert/HateXplain/tree/master/Data) (Mathew et al. 2021). 

For pre-processed datasets and a pre-trained tokenizer used in our experiment, refer to this collection (Insert a Google Drive Link).
<br>After downloading, the pre-trained tokenizer must be inside the folder `model/` and datasets inside the folder `data/` .

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
and predictions will be generated in the same format as the original dataset under the name `WordGRU.pickle`. Again, you can directly download the predictions here (Google Drive link) and place them inside the corresponding folder, e.g., `data/imdb/`

## Model Explainers
Our architecture is described in `explainer.py`. Before training an explainer, 
To train model explainers for each dataset, sequentially run 
```
python main.py config/imdb.json
python main.py config/hatex.json
python main.py config/agnews.json
```

# Evaluation


# Citation
If you use the codes or datasets in this repository, please cite our paper.
