# Deep Learning Skunk Works Projects


### Contents
1. Implementation of Word2Vec using Continuous Bag of Words (CBOW)
1. Intrinsic evaluation metric using analogy labels from ["Efficient Estimation of Word Representations in Vector Space"](https://arxiv.org/pdf/1301.3781.pdf)
1. Implementation of Word2Vec using Skip-gram
   
### Next up
1. TODO: TSNE for visualizing embeddings
1. TODO: GLOVE embeddings
1. TODO: Extrinsic model evaluation
1. TODO: Write unit tests for model training and inference on small data

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
