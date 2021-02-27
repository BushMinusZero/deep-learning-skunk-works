# Deep Learning Skunk Works Projects


### Contents
1. Implementation of Word2Vec using Continuous Bag of Words (CBOW)
1. Intrinsic evaluation metric using analogy labels from ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/pdf/1301.3781.pdf)
1. TODO: Implementation of Word2Vec using Skip-gram
1. TODO: GLOVE embeddings
1. TODO: TSNE for visualizing embeddings
1. TODO: Extrinsic model evaluation


### Training models
- Launch tensorboard
    ```shell
    tensorboard --logdir=data/word2vec/models/
    ```
- Train model
    ```shell
    python src/main.py --train
    ```
- Evaluate model
    ```shell
    python src/main.py --eval
    ```
