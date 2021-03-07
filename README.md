# Deep Learning Skunk Works Projects


### Contents
1. [Implementation of Word2Vec using Continuous Bag of Words (CBOW)](src/cbow.py)
1. [Intrinsic evaluation metric using analogy labels](src/evaluation.py) from ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/pdf/1301.3781.pdf)
1. [Implementation of Word2Vec using Skip-gram](src/skipgram.py)
1. [TSNE for visualizing embeddings of analogy pairs](scripts/Comparing%20t-SNE.ipynb)

### Next up
1. TODO: k-nearest neighbors analysis for finding similar words
1. TODO: filter to the N most common words in the training corpus and mark the rest as OOV
1. TODO: download a larger dataset (GloVe paper uses Gigaword5, Wikipedia2014, and [Common Crawl](https://commoncrawl.org/the-data/get-started/))
1. TODO: Train [GloVe embeddings](https://nlp.stanford.edu/pubs/glove.pdf)
1. TODO: increase the size of the context vector to 300 depending on training speed
1. TODO: Evaluate on word similarity task [WordSim-353](http://alfonseca.org/eng/research/wordsim353.html) used in GloVe paper
1. TODO: Extrinsic model evaluation (NER)
1. TODO: Write unit tests for model training and inference on small data

### Setup
- Developed using Python 3.9 but probably works on Python 3 version
```shell
cd deep-learning-skunk-works/
export PYTHONPATH=`pwd`
export PROJECT_ROOT=`pwd`
pip install -r requirements.txt
```

### Training models
- Set model name (e.g. cbow, skipgram, ...)
    ```shell
    export MODEL='cbow'
    ```
- Launch tensorboard
    ```shell
    tensorboard --logdir=data/$MODEL/models/
    ```
- Train model
    ```shell
    python src/main.py --train --model $MODEL
    ```
- Evaluate model
    ```shell
    python src/main.py --eval --model $MODEL
    ```
