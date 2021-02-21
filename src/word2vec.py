import csv
import os
from datetime import datetime
from typing import Tuple, Generator, List, Optional, Dict

import torch
from torch import Tensor
from torch.nn.functional import relu, log_softmax
from torchtext import data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import torch.nn as nn
import pandas as pd
import numpy as np
from torchtext.data import TabularDataset, BucketIterator, get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm

from src.early_stopping import EarlyStopping
from src.utils import write_losses


class Word2VecConfig:
  batch_size = 32
  embedding_dim = 64
  num_epochs = 10
  patience = 4
  learning_rate_decay = 0.9
  learning_rate_step_size = 1
  context_size = 4

  raw_data_path = os.path.join('data', 'eng.txt')
  output_dir = os.path.join('data', 'word2vec')
  data_dir = os.path.join(output_dir, 'data')
  features_dir = os.path.join(output_dir, 'features')
  model_root_dir = os.path.join(output_dir, 'models')

  def __init__(self, model_date: Optional[str] = None):
    now = (model_date if model_date else datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
    self.model_checkpoint_dir = os.path.join(self.model_root_dir, now)
    os.makedirs(self.output_dir, exist_ok=True)
    os.makedirs(self.model_checkpoint_dir, exist_ok=True)
    self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir, 'model.pt')
    self.model_vocab_path = os.path.join(self.model_checkpoint_dir, 'vocab.txt')


def train_val_test_split(data_path: str, output_dir: str) -> None:
  """Create train/val/test split from randomized input data and write to an output directory."""
  df = pd.read_csv(data_path, sep='\t')
  train_df, val_df, test_df = np.split(df.sample(frac=1), [int(.6 * len(df)), int(.8 * len(df))])

  print(f'Train samples: {len(train_df):,}')
  print(f'Validate samples: {len(val_df):,}')
  print(f'Test samples: {len(test_df):,}')

  train_df.to_csv(os.path.join(output_dir, 'train.tsv'), sep='\t', index=False)
  val_df.to_csv(os.path.join(output_dir, 'val.tsv'), sep='\t', index=False)
  test_df.to_csv(os.path.join(output_dir, 'test.tsv'), sep='\t', index=False)


def load_dataset(output_dir: str) -> Tuple[Tuple[TabularDataset, TabularDataset, TabularDataset], Vocab]:
  """Load train, val, and test datasets."""
  target_text = data.LabelField(lower=True)
  context_text = data.Field(lower=True)
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
  target_text.build_vocab(train.target, train.context, val.target, val.context,
                          specials=['<pad>', '<unk>'])
  context_text.vocab = target_text.vocab
  return (train, val, test), target_text.vocab


def get_device() -> torch.device:
  """Get the device depending on the system's compute hardware."""
  return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def extract_eng_from_parallel_text_file() -> None:
  """Extract just the english text from a TSV file containing eng-fra parallel corpus text."""
  parallel_text_path = 'data/eng-fra.txt'
  df = pd.read_csv(parallel_text_path, sep='\t')
  df.columns = ['eng', 'fra']
  df[['eng']].to_csv('data/eng.txt', index=False)


class DataManager:

  def __init__(self, batch_size: int):
    self.batch_size = batch_size
    self.train_size = None
    self.val_size = None
    self.test_size = None
    self.train_ds, self.val_ds, self.test_ds = None, None, None
    self.train_iterator, self.val_iterator, self.test_iterator = None, None, None
    self.source_vocab = None

  def load_data(self, input_dir: str):
    datasets, self.source_vocab = load_dataset(input_dir)
    (self.train_ds, self.val_ds, self.test_ds) = datasets
    self.train_iterator, self.val_iterator, self.test_iterator = self.get_iterators(
      (self.train_ds, self.val_ds, self.test_ds)
    )
    self.train_size = len(self.train_ds)
    self.val_size = len(self.val_ds)
    self.test_size = len(self.test_ds)

  def get_iterators(self, datasets: Tuple[TabularDataset, TabularDataset, TabularDataset]
                    ) -> Tuple[BucketIterator, BucketIterator, BucketIterator]:
    return data.BucketIterator.splits(
      datasets,
      # shuffle samples between epochs
      shuffle=True,
      device=get_device(),
      batch_sizes=(self.batch_size, self.batch_size, self.batch_size),
      sort_key=lambda x: len(x.context)
    )

  def save_vocab(self, output_path: str) -> None:
    """Save vocabulary generated during model training."""
    with open(output_path, 'w+') as f:
      for token, index in self.source_vocab.stoi.items():
        f.write(f'{index}\t{token}\n')

  def read_vocab(self, path: str):
    vocab = dict()
    with open(path, 'r') as f:
      for line in f:
        index, token = line.split('\t')
        vocab[token.strip()] = int(index)
    return vocab


def create_cbow_features(input_dir: str, output_dir: str) -> None:
  """Creates CBOW features by identifying surrounding context words."""
  # window_size is the number of words to the left and right
  for filename in ['train.tsv', 'val.tsv', 'test.tsv']:
    window_size = 2
    with open(os.path.join(input_dir, filename)) as f_in, \
        open(os.path.join(output_dir, filename), 'w') as f_out:
      reader = csv.reader(f_in, delimiter='\t')
      writer = csv.writer(f_out, delimiter='\t')
      next(reader)  # skip the header
      lines = [line[0].strip().split() for line in reader]
      for words in lines:
        for i, target in enumerate(words):
          words_to_the_left = words[max(0, i - window_size):i]
          words_to_the_right = words[i + 1:i + window_size + 1]
          padded_left = ['<pad>'] * (2 - len(words_to_the_left)) + words_to_the_left
          padded_right = words_to_the_right + ['<pad>'] * (2 - len(words_to_the_right))
          context = padded_left + padded_right
          assert len(context) == window_size * 2
          writer.writerow([target, ' '.join(context)])


def word2vec_preprocessing(raw_data_path: str, data_dir: str, features_dir: str):
  # Split into train/val/test
  train_val_test_split(raw_data_path, data_dir)

  # Tokenize files in data_dir
  # Download en tokenizer with `python -m spacy download en`
  en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  tokenized_rows = []
  for filename in ['test.tsv', 'val.tsv', 'train.tsv']:
    data_path = os.path.join(data_dir, filename)
    with open(data_path) as f:
      reader = csv.reader(f, delimiter='\t')
      next(reader)  # skip header
      for row in reader:
        tokenized_rows.append([' '.join(en_tokenizer(row[0]))])
    with open(data_path, 'w') as f:
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(['eng'])
      writer.writerows(tokenized_rows)

  # Create features
  create_cbow_features(data_dir, features_dir)


class Word2VecModel(nn.Module):
  def __init__(self, vocab_size: int, embedding_dim: int, context_size: int,
               padding_idx: int = 1):
    """ Word to Vec
    :param vocab_size: equal to the size of the vocabulary
    :param embedding_dim: length of the embedding vector for each word
    :param padding_idx:
    """
    super(Word2VecModel, self).__init__()
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

  def __init__(self, config: Word2VecConfig):
    self.config = config
    # Load data and compute vocabulary
    # TODO: remove sentences that exceed the maximum sequence length
    self.data_manager = DataManager(self.config.batch_size)
    self.data_manager.load_data(self.config.features_dir)
    self.data_manager.save_vocab(self.config.model_vocab_path)

    # Initialize model
    self.model = Word2VecModel(
      vocab_size=len(self.data_manager.source_vocab),
      embedding_dim=self.config.embedding_dim,
      context_size=config.context_size
    )

    # Initialize optimizer
    # TODO: move LR to config
    self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=1e-4)

    # Learning rate scheduler
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.config.learning_rate_step_size,
                                               gamma=self.config.learning_rate_decay)

    # Initialize loss function
    self.train_losses = []
    self.val_losses = []
    self.test_losses = []

    self.num_epochs = self.config.num_epochs

    # Initialize early stopping criteria
    self.early_stopping = EarlyStopping(
      model_path=self.config.model_checkpoint_path,
      patience=self.config.patience
    )

    # update learning rate
    print(f"Learning rate: {self.scheduler.get_lr()[0]}")
    self.scheduler.step()

    # initialize tensorboard writer
    self.writer = SummaryWriter(self.config.model_checkpoint_dir)

  def save_model_checkpoint(self, epoch: int):
    torch.save({
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'train_loss': self.train_losses[-1],
      'val_loss': self.val_losses[-1],
    }, self.config.model_checkpoint_path)

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

  def iterate_train(self, epoch: int) -> None:
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
      self.writer.add_scalar('Loss/train', epoch_train_loss, epoch)
      self.writer.add_scalar('Avg Loss/train', epoch_train_loss / self.data_manager.train_size,
                             epoch)

    # Record this epochs total loss
    self.train_losses.append(epoch_train_loss)

  def compute_loss(self, data_iterator: BucketIterator, losses: List[int]) -> float:
    self.model.eval()
    epoch_loss = 0
    for batch in tqdm(data_iterator):
      # Do a forward pass and compute the loss
      loss = self.model.compute_loss(batch.context, batch.target)
      epoch_loss += loss.item()
    losses.append(epoch_loss)
    return epoch_loss

  def compute_validation_loss(self, epoch: int) -> None:
    loss = self.compute_loss(self.data_manager.val_iterator, self.val_losses)
    self.writer.add_scalar('Loss/val', loss, epoch)
    self.writer.add_scalar('Avg Loss/val', loss / self.data_manager.val_size, epoch)

  def compute_test_loss(self) -> None:
    self.compute_loss(self.data_manager.test_iterator, self.test_losses)

  def train(self):
    for epoch in self.iterate_epoch():
      # Forward and backward pass on training data
      self.iterate_train(epoch)

      # Forward pass on validation data
      self.compute_validation_loss(epoch)

    # Compute test loss
    # TODO: can we compute test loss? need to handle missing values? map them to UNK?
    # self.compute_test_loss()


def train():
  conf = Word2VecConfig()
  word2vec_preprocessing(conf.raw_data_path, conf.data_dir, conf.features_dir)
  loop = TrainingLoop(conf)
  loop.train()
  write_losses(os.path.join(conf.model_checkpoint_dir, 'losses.csv'),
               loop.train_losses, loop.val_losses)


class InferenceServer:

  def __init__(self, config: Word2VecConfig):
    self.config = config
    self.model = None
    data_manager = DataManager(self.config.batch_size)
    self.vocab = data_manager.read_vocab(self.config.model_vocab_path)
    self.unk_index = self.vocab['<unk>']

  def embed_tokens(self, tokens: List[str]) -> Tensor:
    sentence_indices = [self.vocab.get(word, self.unk_index) for word in tokens]
    sentence_tensor = torch.LongTensor(sentence_indices)
    return self.model.embeddings(sentence_tensor)

  def load_model(self, model_path: str):
    self.model = Word2VecModel(
      vocab_size=len(self.vocab),
      embedding_dim=self.config.embedding_dim,
      context_size=self.config.context_size
    )
    model_info = torch.load(model_path)
    self.model.load_state_dict(model_info['model_state_dict'])
    self.model.eval()


def embed_test_data():
  # TODO: get latest model date form models dir
  conf = Word2VecConfig(model_date='2021-02-20T15:42:57')
  server = InferenceServer(conf)
  server.load_model(conf.model_checkpoint_path)

  with open(os.path.join(conf.data_dir, 'test.tsv')) as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # skip header
    for row in reader:
      server.model.embedding(row[0].split())
    # TODO: save embedding predictions


def inference():
  # TODO: get latest model date form models dir
  conf = Word2VecConfig(model_date='2021-02-20T17:45:19')
  server = InferenceServer(conf)
  server.load_model(conf.model_checkpoint_path)

  word1 = 'hello'
  word2 = 'hi'
  word1_index = server.vocab.get(word1, None)
  word2_index = server.vocab.get(word2, None)
  print(f'Word1 index: {word1_index}')
  print(f'Word2 index: {word2_index}')

  e1 = server.embed_tokens([word1])
  e2 = server.embed_tokens([word2])
  # breakpoint()
  l2_norm = torch.norm(torch.dot(e1.squeeze(), e2.squeeze()))
  print(l2_norm.item())
  # TODO: compare words


if __name__ == '__main__':
  train()
  # inference()
