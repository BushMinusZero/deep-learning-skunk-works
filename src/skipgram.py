import csv
import os
import random
from typing import Dict, Tuple, List, Optional, Callable
from collections import Counter
import math

from torch import Tensor, LongTensor, optim
from torchtext import data
import torch.utils.data

from src.config import SkipGramConfig
from src.models.skipgram import SkipGramModel
from src.train_model import TrainingLoop
from src.utils import train_val_test_split, tokenize_english_text, write_losses


def get_tokenizer(language: str):
  if language == 'en':
    return data.get_tokenizer('spacy', language='en_core_web_sm')
  else:
    raise ValueError("Only english tokenization is currently supported.")


def build_counter_stoi_and_itos(text_path: str) -> Tuple[Counter, Dict[str, int], Dict[int, str]]:
  """Assumes that text is already tokenized."""
  token_counter = Counter()
  with open(text_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
      tokens = row[0].split()
      token_counter += Counter(tokens)
  stoi = {t: i for i, t in enumerate(token_counter)}
  itos = {i: t for t, i in stoi.items()}
  return token_counter, stoi, itos


def create_training_examples(data_path: str, counter: Counter, oov_token: str,
                             context_size: int) -> List[Tuple[str, str]]:
  """Creates training example while subsampling frequent words.
  Assumes that text is already tokenized."""
  total_token_count = sum(counter.values())
  examples = []
  window_size = context_size // 2
  with open(data_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
      sentence = row[0].split()
      kept_words = []
      for word in sentence:
        word_count = counter.get(word, oov_token)
        word_fraction = word_count / total_token_count
        prob_of_keeping_word = (math.sqrt(word_fraction / 0.001) + 1) * (0.001 / word_fraction)
        if random.random() < prob_of_keeping_word:
          # Randomly dropout words that with a probability related to a terms frequency
          kept_words.append(word)
      for i, target in enumerate(sentence):
        words_to_the_left = sentence[max(0, i - window_size):i]
        words_to_the_right = sentence[i + 1:i + window_size + 1]
        for context in words_to_the_left + words_to_the_right:
          examples.append((target, context))
  return examples


class SkipGramDataset(torch.utils.data.Dataset):

  def __init__(self, data_path: str, context_size: int):
    self.token_counter, self.stoi, self.itos = build_counter_stoi_and_itos(data_path)
    self.oov = '<oov>'
    self.pad = '<pad>'
    self.examples = create_training_examples(data_path, self.token_counter, self.oov,
                                             context_size)

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx: int) -> Tuple[Tuple[int, int], List[int]]:
    example = self.examples[idx]
    return self.generate_negative_samples(example, num_negative_samples=10)

  def generate_negative_samples(self, example: Tuple[str, str],
                                num_negative_samples: int) -> Tuple[Tuple[int, int], List[int]]:
    """Return example with negative samples."""
    example_indices = (self.stoi[example[0]], self.stoi[example[1]])
    neg_sample_indices = [random.randint(0, len(self.stoi) - 1)
                          for _ in range(num_negative_samples)]
    return example_indices, neg_sample_indices

  def save_vocab(self, output_path: str) -> None:
    """Save vocabulary generated during model training."""
    with open(output_path, 'w+') as f:
      for token, index in self.stoi.items():
        f.write(f'{index}\t{token}\n')


class SkipGramDataLoader(torch.utils.data.DataLoader):

  def __init__(self, dataset: SkipGramDataset, batch_size: int = None, shuffle: bool = False,
               collate_fn: Optional[Callable] = None):
    collate_fn = collate_fn if collate_fn else self.collate_examples
    super().__init__(dataset, batch_size, shuffle, collate_fn=collate_fn)

  @staticmethod
  def collate_examples(batch: List[Tuple[Tuple[int, int], List[int]]]
                       ) -> Tuple[Tensor, Tensor, Tensor]:
    """Collate example."""
    target_context_pair, negative_samples = zip(*batch)
    target, context = zip(*target_context_pair)
    return LongTensor(target), LongTensor(context), LongTensor(negative_samples)


class SkipGramTrainingLoop(TrainingLoop):

  def compute_loss_from_batch(self, batch) -> Tensor:
    # Do a forward pass and compute the loss
    return self.model.forward(batch[0], batch[1], batch[2])


def train():
  config = SkipGramConfig()

  # Split into train/val/test
  train_val_test_split(config.raw_data_path, config.data_dir)

  # Tokenize text
  for filename in ['test.tsv', 'val.tsv', 'train.tsv']:
    data_path = os.path.join(config.data_dir, filename)
    tokenize_english_text(data_path, data_path)

  # Training data
  training_data_path = os.path.join(config.data_dir, 'train.tsv')
  train_dataset = SkipGramDataset(training_data_path, config.context_size)
  train_data_loader = SkipGramDataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
  model = SkipGramModel(len(train_dataset.stoi), config.embedding_dim)

  # Validation data
  val_data_path = os.path.join(config.data_dir, 'val.tsv')
  val_dataset = SkipGramDataset(val_data_path, config.context_size)
  val_data_loader = SkipGramDataLoader(val_dataset, shuffle=False, batch_size=config.batch_size)

  # Initialize optimizer
  # TODO: move LR to config
  optimizer = optim.SparseAdam(list(model.parameters()), lr=0.001)

  loop = SkipGramTrainingLoop(config, model, optimizer, train_data_loader, val_data_loader)
  loop.train()
  write_losses(os.path.join(config.model_checkpoint_dir, 'losses.csv'),
               loop.train_losses, loop.val_losses)


if __name__ == '__main__':
  train()
