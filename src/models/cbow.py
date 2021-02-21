import torch
from torch import nn, Tensor, log_softmax


class CBOWModel(nn.Module):
  def __init__(self, vocab_size: int, embedding_dim: int, context_size: int):
    """ Word to Vector model
    :param vocab_size: equal to the size of the vocabulary
    :param embedding_dim: length of the embedding vector for each word
    :param context_size: the number of context tokens surrounding the target token
    """
    super(CBOWModel, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.context_size = context_size
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # (vocab_size, 64)
    self.linear1 = nn.Linear(context_size * embedding_dim, 128)  # (64x4, 128) -> (256, 128)
    self.linear2 = nn.Linear(128, vocab_size)
    self.loss_function = nn.NLLLoss()

  def forward(self, inputs: Tensor) -> Tensor:
    embeds = self.embeddings(inputs)  # output is batch_size x context size x embedding dim
    batch_size = inputs.shape[0]  # find the batch size on the fly because partial batches may exist
    embeds = embeds.view((batch_size, -1))  # stacks the context word vectors in each batch
    out = torch.relu(self.linear1(embeds))
    out = self.linear2(out)
    log_probabilities = log_softmax(out, dim=1)
    return log_probabilities

  def compute_loss(self, context: Tensor, target: Tensor) -> Tensor:
    """Computes negative log likelihood loss."""
    # Transpose batch because embedding in forward() expects N x X (batch_size by sequence length)
    context = torch.transpose(context, 0, 1)

    # Run the forward pass, getting log probabilities over next batch of words
    log_probabilities = self.forward(context)

    # Compute your loss function.
    target = target.squeeze()  # (1, batch size) -> (batch size)
    return self.loss_function(log_probabilities, target)
