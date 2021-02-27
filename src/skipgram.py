import csv
import os
from typing import Dict, Tuple, List
from collections import Counter
from torchtext.data import get_tokenizer

from src.config import SkipGramConfig
from src.models.skipgram import SkipGramModel
from src.utils import train_val_test_split


class Vocab:

  def __init__(self, language: str):
    self.language = language
    self.tokenizer = self.get_tokenizer(language)
    self.stoi, self.itos = None, None

  @staticmethod
  def get_tokenizer(language: str):
    if language == 'en':
      return get_tokenizer('spacy', language='en_core_web_sm')
    else:
      raise ValueError("Only english tokenization is currently supported.")

  def build_stoi_and_itos(self, text_path: str) -> None:
    token_counter = Counter()
    with open(text_path) as f:
      reader = csv.reader(f)
      next(reader)  # skip header
      for row in reader:
        tokens = self.tokenizer(row[0])
        token_counter += Counter(tokens)
    # TODO: find and remove stopwords with tf-idf

    self.stoi = {t: i for i, t in enumerate(token_counter)}
    self.itos = {i: t for t, i in self.stoi.items()}

  def save_vocab(self, output_path: str) -> None:
    """Save vocabulary generated during model training."""
    with open(output_path, 'w+') as f:
      for token, index in self.stoi.items():
        f.write(f'{index}\t{token}\n')

  def __len__(self):
    return len(self.stoi)


def combine_vocabs(vocab1: Vocab, vocab2: Vocab) -> Vocab:
  vocab = Vocab(vocab1.language)
  vocab.stoi = {**vocab1.stoi, **vocab2.stoi}
  vocab.itos = {i: t for t, i in vocab.stoi.items()}
  return vocab


class DataManager:

  def __init__(self, data_dir: str, language: str):
    self.data_dir = data_dir
    self.language = language
    self.train_vocab, self.val_vocab, self.test_vocab = self.load_vocabs()

  def load_vocabs(self) -> List[Vocab]:
    vocabs = []
    for filename in ['train.tsv', 'val.tsv', 'test.tsv']:
      path = os.path.join(self.data_dir, filename)
      vocab = Vocab(self.language)
      vocab.build_stoi_and_itos(path)
      vocabs.append(vocab)
    return vocabs


def train():
  config = SkipGramConfig()

  # Split into train/val/test
  # train_val_test_split(config.raw_data_path, config.data_dir)

  # Build Vocabularies of string-to-index and index-to-string maps
  data_manager = DataManager(config.data_dir, language='en')
  vocab = combine_vocabs(data_manager.train_vocab, data_manager.val_vocab)
  vocab_size = len(vocab)
  model = SkipGramModel(vocab_size, config.embedding_dim)

  # TODO: Create train, and val data iterators with negative sampling
  # TODO: Add training loop
  # TODO: Evaluation metrics


if __name__ == '__main__':
  train()
