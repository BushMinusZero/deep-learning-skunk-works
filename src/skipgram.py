import csv
import os
import random
from typing import Dict, Tuple, List, Optional, Callable, Union
from collections import Counter
import math

import numpy as np
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


def build_counter_stoi_and_itos(text_path: str, special_tokens: List[str]
                                ) -> Tuple[Counter, Dict[str, int], Dict[int, str]]:
  """Assumes that text is already tokenized."""
  token_counter = Counter()
  token_counter.update({token: 1 for token in special_tokens})
  with open(text_path) as f:
    reader = csv.reader(f)
    next(reader)  # skip header
    for row in reader:
      tokens = row[0].split()
      token_counter += Counter(tokens)
  stoi = {t: i for i, t in enumerate(token_counter)}
  itos = {i: t for t, i in stoi.items()}
  return token_counter, stoi, itos


def create_training_examples(data_path: str, counter: Union[Counter, Dict[str, int]],
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
        word_count = counter.get(word, 1)
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


def generate_neg_sample_indices(num_negative_samples: int, stoi: Dict[str, int],
                                token_counter: Union[Counter, Dict[str, int]],
                                num_examples: int) -> List[int]:
  """Randomly generates `num_negative_samples` for each training example."""
  token_indices = [stoi[t] for t in token_counter.keys()]
  weighted_token_weights = [math.pow(t, 3 / 4) for t in token_counter.values()]
  # Remove the OOV token
  token_indices, weighted_token_weights = token_indices[1:], weighted_token_weights[1:]
  weighted_token_weights = [t / sum(weighted_token_weights) for t in weighted_token_weights]
  return np.random.choice(token_indices, p=weighted_token_weights,
                          size=num_examples * num_negative_samples)


class SkipGramTrainingDataset(torch.utils.data.Dataset):

  def __init__(self, data_path: str, context_size: int, oov_token: str = '<oov>'):
    self.token_counter, self.stoi, self.itos = build_counter_stoi_and_itos(data_path, [oov_token])
    self.examples = create_training_examples(data_path, self.token_counter, context_size)
    self.num_negative_samples = 10
    self.neg_sample_indices = generate_neg_sample_indices(self.num_negative_samples, self.stoi,
                                                          self.token_counter, len(self.examples))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx: int) -> Tuple[Tuple[int, int], List[int]]:
    example = self.examples[idx]
    example_indices = (self.stoi[example[0]], self.stoi[example[1]])
    neg_index = idx * self.num_negative_samples
    return example_indices, self.neg_sample_indices[neg_index:neg_index+self.num_negative_samples]

  def save_vocab(self, output_path: str) -> None:
    """Save vocabulary generated during model training."""
    with open(output_path, 'w+') as f:
      writer = csv.writer(f, delimiter='\t')
      writer.writerow(['index', 'token', 'count'])
      for token, index in self.stoi.items():
        writer.writerow([index, token, self.token_counter[token]])


def load_counter_stoi_and_itos(vocab_path: str) -> Tuple[Dict[str, int], Dict[str, int],
                                                         Dict[int, str]]:
  counter, stoi, itos = {}, {}, {}
  with open(vocab_path) as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)  # skip header
    for row in reader:
      index, token, count = int(row[0]), str(row[1]), int(row[2])
      counter[token] = count
      itos[index] = token
      stoi[token] = index
  return counter, stoi, itos


class SkipGramValidationDataset(torch.utils.data.Dataset):

  def __init__(self, data_path: str, context_size: int, vocab_path: str):
    self.token_counter, self.stoi, self.itos = load_counter_stoi_and_itos(vocab_path)
    self.oov_index = self.stoi['<oov>']
    self.examples = create_training_examples(data_path, self.token_counter, context_size)
    self.num_negative_samples = 10
    self.neg_sample_indices = generate_neg_sample_indices(self.num_negative_samples, self.stoi,
                                                          self.token_counter, len(self.examples))

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, idx: int) -> Tuple[Tuple[int, int], List[int]]:
    example = self.examples[idx]
    example_indices = (self.stoi.get(example[0], self.oov_index),
                       self.stoi.get(example[1], self.oov_index))
    neg_index = idx * self.num_negative_samples
    return example_indices, self.neg_sample_indices[neg_index:neg_index + self.num_negative_samples]


class SkipGramDataLoader(torch.utils.data.DataLoader):

  def __init__(self, dataset: Union[SkipGramValidationDataset, SkipGramTrainingDataset],
               batch_size: int = None, shuffle: bool = False,
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
  train_dataset = SkipGramTrainingDataset(training_data_path, config.context_size)
  train_data_loader = SkipGramDataLoader(train_dataset, shuffle=True, batch_size=config.batch_size)
  train_dataset.save_vocab(config.model_vocab_path)
  model = SkipGramModel(len(train_dataset.stoi), config.embedding_dim)

  # Validation data
  val_data_path = os.path.join(config.data_dir, 'val.tsv')
  val_dataset = SkipGramValidationDataset(val_data_path, config.context_size,
                                          config.model_vocab_path)
  val_data_loader = SkipGramDataLoader(val_dataset, shuffle=False, batch_size=config.batch_size)

  # Initialize optimizer
  # TODO: move LR to config
  optimizer = optim.SparseAdam(list(model.parameters()), lr=0.001)

  loop = SkipGramTrainingLoop(config, model, optimizer, train_data_loader, val_data_loader)
  loop.train()
  write_losses(os.path.join(config.model_checkpoint_dir, 'losses.csv'),
               loop.train_losses, loop.val_losses)


def load_skip_gram_model(model_path: str, vocab: Dict[str, int], config: SkipGramConfig):
  model = SkipGramModel(
    vocab_size=len(vocab),
    embedding_dim=config.embedding_dim
  )
  model_info = torch.load(model_path)
  model.load_state_dict(model_info['model_state_dict'])
  model.eval()
  return model
