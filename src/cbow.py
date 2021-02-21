import csv
import os
from datetime import datetime
from typing import Tuple, Generator, List, Callable

import torch
from torch import Tensor
from torchtext import data
from torch.utils.tensorboard import SummaryWriter
import torch.optim as optim
import pandas as pd
from torchtext.data import TabularDataset, BucketIterator, get_tokenizer
from torchtext.vocab import Vocab
from tqdm import tqdm

from src.config import CBOWConfig
from src.early_stopping import EarlyStopping
from src.models.cbow import CBOWModel
from src.utils import write_losses, cosine_similarity, train_val_test_split


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

  @staticmethod
  def read_vocab(path: str):
    vocab = dict()
    with open(path, 'r') as f:
      for line in f:
        index, token = line.split('\t')
        vocab[token.strip()] = int(index)
    return vocab


def create_cbow_features(input_dir: str, output_dir: str, context_size: int) -> None:
  """Creates CBOW features by identifying surrounding context words."""
  assert context_size % 2 == 0, "Context size should be even so we pick the same number of words" \
                                " to the left and right of the target word."
  # window_size is the number of words to the left and right
  for filename in ['train.tsv', 'val.tsv', 'test.tsv']:
    window_size = context_size // 2
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
          padded_left = ['<pad>'] * (window_size - len(words_to_the_left)) + words_to_the_left
          padded_right = words_to_the_right + ['<pad>'] * (window_size - len(words_to_the_right))
          context = padded_left + padded_right
          assert len(context) == window_size * 2
          writer.writerow([target, ' '.join(context)])


def cbow_preprocessing(config: CBOWConfig):
  # Split into train/val/test
  train_val_test_split(config.raw_data_path, config.data_dir)

  # Tokenize files in data_dir
  # Download en tokenizer with `python -m spacy download en`
  en_tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
  tokenized_rows = []
  for filename in ['test.tsv', 'val.tsv', 'train.tsv']:
    data_path = os.path.join(config.data_dir, filename)
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
  create_cbow_features(config.data_dir, config.features_dir, config.context_size)


class TrainingLoop:

  def __init__(self, config: CBOWConfig):
    self.config = config
    # Load data and compute vocabulary
    # TODO: remove sentences that exceed the maximum sequence length
    self.data_manager = DataManager(self.config.batch_size)
    self.data_manager.load_data(self.config.features_dir)
    self.data_manager.save_vocab(self.config.model_vocab_path)

    # Initialize model
    self.model = CBOWModel(
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


def train():
  conf = CBOWConfig()
  cbow_preprocessing(conf)
  loop = TrainingLoop(conf)
  loop.train()
  write_losses(os.path.join(conf.model_checkpoint_dir, 'losses.csv'),
               loop.train_losses, loop.val_losses)


# TODO: make a generic embedding inference server general to any pytorch model + vocab
class InferenceServer:

  def __init__(self, config: CBOWConfig):
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
    self.model = CBOWModel(
      vocab_size=len(self.vocab),
      embedding_dim=self.config.embedding_dim,
      context_size=self.config.context_size
    )
    model_info = torch.load(model_path)
    self.model.load_state_dict(model_info['model_state_dict'])
    self.model.eval()

  def embed_word(self, word: str) -> Tensor:
    return self.embed_tokens([word]).squeeze()

  def calculate_similarity(self, word1: str, word2: str,
                           similarity_func: Callable = cosine_similarity) -> float:
    e1 = self.embed_word(word1)
    e2 = self.embed_word(word2)
    return similarity_func(e1, e2).item()


def get_latest_cbow_model(date_fmt: str = '%Y-%m-%dT%H:%M:%S') -> str:
  model_dirs = os.listdir(CBOWConfig.model_root_dir)
  dates = [datetime.strptime(d, date_fmt) for d in model_dirs]
  return max(dates).strftime(date_fmt)


def inference():
  conf = CBOWConfig(model_date=get_latest_cbow_model())
  server = InferenceServer(conf)
  server.load_model(conf.model_checkpoint_path)

  word1, word2 = 'hello', 'hi'
  similarity = server.calculate_similarity(word1, word2, similarity_func=cosine_similarity)
  print(f'Similarity between {word1} and {word2} is {similarity}')
