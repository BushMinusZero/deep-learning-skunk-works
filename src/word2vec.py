import csv
import os
from typing import Tuple, Generator

import torch
from torch import Tensor
from torch.nn.functional import relu, log_softmax
from torchtext import data
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from torchtext.data import TabularDataset, BucketIterator
from torchtext.vocab import Vocab
from tqdm import tqdm

from src.early_stopping import EarlyStopping


class Word2VecConstants:
  raw_data_path = os.path.join('data', 'eng.txt')
  output_dir = os.path.join('data', 'word2vec')
  data_dir = os.path.join(output_dir, 'data')
  features_dir = os.path.join(output_dir, 'features')
  model_checkpoint_dir = os.path.join(output_dir, 'model')
  os.makedirs(output_dir, exist_ok=True)
  os.makedirs(model_checkpoint_dir, exist_ok=True)
  model_checkpoint_path = os.path.join(model_checkpoint_dir, 'model.pt')
  batch_size = 32
  embedding_dim = 64
  num_epochs = 10


def train_val_test_split(data_path: str, output_dir: str) -> None:
  """Create train/val/test split from randomized input data and write to an output directory."""
  df = pd.read_csv(data_path, sep='\t')
  train, val, test = np.split(df.sample(frac=1), [int(.6*len(df)), int(.8*len(df))])

  print(f'Train samples: {len(train):,}')
  print(f'Validate samples: {len(val):,}')
  print(f'Test samples: {len(test):,}')

  train.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
  val.to_csv(os.path.join(output_dir, 'val.tsv'), sep='\t', index=False)
  test.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)


def load_dataset(output_dir: str) -> Tuple[Tuple[TabularDataset, TabularDataset, TabularDataset],
                                           Vocab]:
  """Load train, val, and test datasets."""
  target_text = data.LabelField()
  context_text = data.Field()
  fields = [
    ('target', target_text),
    ('context', context_text),
  ]
  train, val, test = data.TabularDataset.splits(
    path=output_dir,
    train='train.tsv',
    validation='val.tsv',
    test='test.tsv',
    format='tsv',
    fields=fields,
    skip_header=True
  )
  target_text.build_vocab(train.target, train.context, val.target, val.context)
  context_text.vocab = target_text.vocab
  return (train, val, test), target_text.vocab


def get_device() -> torch.device:
  """Get the device depending on the system's compute hardware."""
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_iterators(datasets: Tuple[TabularDataset, TabularDataset, TabularDataset]
                  ) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:
  return data.BucketIterator.splits(
    datasets,
    # shuffle samples between epochs
    shuffle=True,
    device=get_device(),
    batch_sizes=(Word2VecConstants.batch_size,
                 Word2VecConstants.batch_size,
                 Word2VecConstants.batch_size),
    sort_key=lambda x: len(x.context)
  )


def extract_eng_from_parallel_text_file() -> None:
  """Extract just the english text from a TSV file containing eng-fra parallel corpus text."""
  parallel_text_path = 'data/eng-fra.txt'
  df = pd.read_csv(parallel_text_path, sep='\t')
  df.columns = ['eng', 'fra']
  df[['eng']].to_csv('data/eng.txt', index=False)


class DataManager:

  def __init__(self, input_dir: str):
    datasets, self.source_vocab = load_dataset(input_dir)
    (self.train_ds, self.val_ds, self.test_ds) = datasets
    self.train_iterator, self.val_iterator, self.test_iterator = get_iterators(
      (self.train_ds, self.val_ds, self.test_ds)
    )


def create_cbow_features(input_dir: str, output_dir: str) -> None:
  """Creates CBOW features by identifying surrounding context words."""
  # window_size is the number of words to the left and right
  for filename in ['train.tsv', 'val.tsv', 'test.tsv']:
    window_size = 2
    with open(os.path.join(input_dir,  filename)) as f_in, \
        open(os.path.join(output_dir, filename), 'w') as f_out:
      reader = csv.reader(f_in, delimiter='\t')
      writer = csv.writer(f_out, delimiter='\t')
      next(reader)  # skip the header
      lines = [line[0].strip().split() for line in reader]
      for words in lines:
        for i, target in enumerate(words):
          words_to_the_left = words[max(0, i-window_size):i]
          words_to_the_right = words[i+1:i+window_size+1]
          padded_left = ['<pad>'] * (2 - len(words_to_the_left)) + words_to_the_left
          padded_right = words_to_the_right + ['<pad>'] * (2 - len(words_to_the_right))
          context = padded_left + padded_right
          assert len(context) == window_size * 2
          writer.writerow([target, ' '.join(context)])


def word2vec_preprocessing():
  # TODO: add data pre-processing steps (tokenization etc.)
  train_val_test_split(Word2VecConstants.raw_data_path, Word2VecConstants.data_dir)

  # Create features
  create_cbow_features(Word2VecConstants.data_dir, Word2VecConstants.features_dir)


class Word2Vec(nn.Module):
  def __init__(self, vocab_size: int, embedding_dim: int, context_size: int,
               padding_idx: int = 1):
    """ Word to Vec
    :param vocab_size: equal to the size of the vocabulary
    :param embedding_dim: length of the embedding vector for each word
    :param padding_idx:
    """
    super(Word2Vec, self).__init__()
    self.vocab_size = vocab_size
    self.embedding_dim = embedding_dim
    self.padding_idx = padding_idx
    self.context_size = context_size
    # TODO: add embedding max_norm
    self.embeddings = nn.Embedding(vocab_size, embedding_dim)  # (vocab_size, 64)
    self.linear1 = nn.Linear(context_size * embedding_dim, 128)  # (64x4, 128) -> (256, 128)
    self.linear2 = nn.Linear(128, vocab_size)
    self.loss_function = nn.NLLLoss()

  def forward(self, inputs: Tensor) -> Tensor:
    embeds = self.embeddings(inputs)  # output is batch_size x context size x embedding dim
    batch_size = inputs.shape[0]  # find the batch size on the fly because partial batches may exist
    embeds = embeds.view((batch_size, -1))  # stacks the context word vectors in each batch
    out = relu(self.linear1(embeds))
    out = self.linear2(out)
    log_probabilities = log_softmax(out, dim=1)
    return log_probabilities

  def compute_loss(self, context: Tensor, target: Tensor) -> Tensor:
    # Transpose batch because embedding in forward() expects N x X (batch_size by sequence length)
    context = torch.transpose(context, 0, 1)
    
    # Run the forward pass, getting log probabilities over next batch of words
    log_probabilities = self.forward(context)

    # Compute your loss function.
    target = target.squeeze()  # (1, batch size) -> (batch size)
    return self.loss_function(log_probabilities, target)


class TrainingLoop:

  def __init__(self):
    # Load data and compute vocabulary
    # TODO: remove sentences that exceed the maximum sequence length
    self.data_manager = DataManager(Word2VecConstants.features_dir)

    # Initialize model
    self.model = Word2Vec(
      vocab_size=len(self.data_manager.source_vocab),
      embedding_dim=Word2VecConstants.embedding_dim,
      context_size=4
    )

    # Initialize optimizer
    # TODO: move LR to constants
    # TODO: configure other optimization parameters
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)

    # Initialize loss function
    self.train_losses = []
    self.val_losses = []

    # TODO: Make constants a init param
    self.num_epochs = Word2VecConstants.num_epochs

    # Initialize early stopping criteria
    # TODO: move patience to constants
    self.early_stopping = EarlyStopping(
      model_path=Word2VecConstants.model_checkpoint_path,
      patience=5
    )

  def save_model_checkpoint(self, epoch: int):
    torch.save({
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'train_loss': self.train_losses[-1],
      'val_loss': self.val_losses[-1],
    }, Word2VecConstants.model_checkpoint_path)

  def iterate_epoch(self) -> Generator[int, None, None]:
    for epoch in range(self.num_epochs):
      print(f'Starting epoch {epoch}')

      # Iterate over the epochs and train model here
      yield epoch

      print(f'Train loss: {self.train_losses[-1]}')
      print(f'Val loss:   {self.val_losses[-1]}')
      self.early_stopping.check_early_stopping_criteria(self.model, self.optimizer, epoch,
                                                        train_loss=self.train_losses[-1],
                                                        val_loss=self.val_losses[-1])
      if self.early_stopping.early_stop:
        # End epoch iteration if we meet the early stopping criteria
        print(f"Triggered early stopping criteria. Stopping after the {epoch}th iteration")
        break

  def iterate_train(self) -> None:
    epoch_train_loss = 0
    self.model.train()
    for batch in tqdm(self.data_manager.train_iterator):
      # batch is of type torchtext.data.batch.Batch
      # batch has dimension X x N where X is the max sequence length and N is the batch_size
      # the values in batch are the indices of the word in data_manager.source_vocab.itos

      # Before passing in a new instance, you need to zero out the gradients from the old instance
      self.model.zero_grad()
      
      # Do a forward pass and compute the loss
      loss = self.model.compute_loss(batch.context, batch.target)

      # Do the backward pass and update the gradient
      loss.backward()
      self.optimizer.step()

      # Update the total loss for the current epoch
      epoch_train_loss += loss.item()

    # Record this epochs total loss
    self.train_losses.append(epoch_train_loss)

  def compute_validation_loss(self) -> None:
    self.model.eval()
    epoch_val_loss = 0
    for batch in tqdm(self.data_manager.val_iterator):
      # Do a forward pass and compute the loss
      loss = self.model.compute_loss(batch.context, batch.target)
      epoch_val_loss += loss.item()
    self.val_losses.append(epoch_val_loss)

  def train(self):
    for _ in self.iterate_epoch():
      # Forward and backward pass on training data
      self.iterate_train()

      # Forward pass on validation data
      self.compute_validation_loss()


if __name__ == '__main__':
  word2vec_preprocessing()
  loop = TrainingLoop()
  loop.train()
  # TODO: Compute loss on test set after training is complete
  # TODO: Evaluate embeddings
  # TODO: Write function to load pre-trained model
  # TODO: Add tensorboard logs
