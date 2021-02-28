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
from src.train_model import TrainingLoop
from src.utils import write_losses, cosine_similarity, train_val_test_split, tokenize_english_text


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
  for filename in ['test.tsv', 'val.tsv', 'train.tsv']:
    data_path = os.path.join(config.data_dir, filename)
    tokenize_english_text(data_path, data_path)

  # Create features
  create_cbow_features(config.data_dir, config.features_dir, config.context_size)


class CBOWTrainingLoop(TrainingLoop):

  def compute_loss_from_batch(self, batch) -> Tensor:
    # Do a forward pass and compute the loss
    return self.model.compute_loss(batch.context, batch.target)


def train():
  conf = CBOWConfig()
  cbow_preprocessing(conf)

  # Load data and compute vocabulary
  # TODO: remove sentences that exceed the maximum sequence length
  data_manager = DataManager(conf.batch_size)
  data_manager.load_data(conf.features_dir)
  data_manager.save_vocab(conf.model_vocab_path)

  # Initialize model
  model = CBOWModel(
    vocab_size=len(data_manager.source_vocab),
    embedding_dim=conf.embedding_dim,
    context_size=conf.context_size
  )

  # Initialize optimizer
  # TODO: move LR to config
  optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)

  loop = CBOWTrainingLoop(conf, model, optimizer, data_manager.train_iterator,
                          data_manager.val_iterator)
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
