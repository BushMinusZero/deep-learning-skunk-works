import torch
from torch import nn, Tensor, log_softmax


class SkipGramModel(nn.Module):
  def __init__(self, vocab_size: int, embedding_dim: int):
    """ Word to Vector Skip Gram model
    :param vocab_size: equal to the size of the vocabulary
    :param embedding_dim: length of the embedding vector for each word
    """
    super(SkipGramModel, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.word_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.init_weights()

  def init_weights(self):
    """Initialize embeddings parameters."""
    init_range = (2.0 / (self.vocab_size + self.embedding_dim)) ** 0.5
    self.word_embeddings.weight.data.uniform_(-init_range, init_range)
    self.context_embeddings.weight.data.uniform_(0, 0)

  def forward(self, target: Tensor, pos_context: Tensor, neg_context: Tensor):
    """Forward pass using negative sampling.
    :param target: target tensor of [batch_size]
    :param pos_context: positive context tensor of [batch_size]
    :param neg_context: negative context tensor of [batch_size, num_negative_samples]
    :return: loss
    """
    # TODO: implement forward pass
    pass
