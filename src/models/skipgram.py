import torch
from torch import nn, Tensor


class SkipGramModel(nn.Module):
  def __init__(self, vocab_size: int, embedding_dim: int):
    """ Word to Vector Skip Gram model
    :param vocab_size: equal to the size of the vocabulary
    :param embedding_dim: length of the embedding vector for each word
    """
    super(SkipGramModel, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.target_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.context_embeddings = nn.Embedding(vocab_size, embedding_dim, sparse=True)
    self.init_weights()
    self.log_sigmoid = nn.LogSigmoid()

  def init_weights(self):
    """Initialize embeddings parameters."""
    init_range = (2.0 / (self.vocab_size + self.embedding_dim)) ** 0.5
    self.target_embeddings.weight.data.uniform_(-init_range, init_range)
    self.context_embeddings.weight.data.uniform_(0, 0)

  def forward(self, target: Tensor, pos_context: Tensor, neg_context: Tensor):
    """Calculate the log loss using negative sampling.
    :param target: target tensor of [batch_size]
    :param pos_context: positive context tensor of [batch_size]
    :param neg_context: negative context tensor of [batch_size, num_negative_samples]
    :return: negative log loss
    """
    # Embed all the things
    # [batch_size]
    embedded_target = self.target_embeddings(target)
    # [batch_size]
    embedded_pos_context = self.context_embeddings(pos_context)
    # [batch_size, num_negative_samples]
    embedded_neg_context = self.context_embeddings(neg_context)

    # Compute loss of positive context
    # [batch_size]
    pos_loss = self.log_sigmoid(torch.sum(embedded_target * embedded_pos_context))

    # Compute loss of negative context using bmm (batch matrix-matrix) product
    # [batch_size, num_negative_samples, emb_dim] x [batch_size, emb_dim, 1]
    neg_val = torch.bmm(embedded_neg_context, embedded_target.unsqueeze(2))
    # [batch_size]
    neg_loss = self.log_sigmoid(-torch.sum(neg_val, dim=1)).squeeze()

    # Sum and return losses
    return -1 * (pos_loss + neg_loss).sum()

  def predict(self, target: Tensor):
    """Make an inference using the target embedding."""
    return self.target_embeddings(target)
