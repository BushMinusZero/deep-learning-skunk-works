import csv
import os
import unicodedata
import re
from typing import Tuple, List
import time
import math

import torch
from torch import Tensor


class LanguageDictionary:
    """Collects word metadata for building one-hot-encoding vector."""
    SOS_token = 0
    EOS_token = 1

    def __init__(self, name: str):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {LanguageDictionary.SOS_token: 'SOS', LanguageDictionary.EOS_token: 'EOS'}
        self.n_words = 2

    def add_word(self, word: str):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def add_sentence(self, sentence: str):
        # TODO: improve tokenization
        for word in sentence.split(' '):
            self.add_word(word)


def unicode_to_ascii(s):
  """Turn a Unicode string to plain ASCII, thanks to
  https://stackoverflow.com/a/518232/2809427
  """
  return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )


def normalize_string(s):
  """Lowercase, trim, and remove non-letter characters"""
  s = unicode_to_ascii(s.lower().strip())
  s = re.sub(r"([.!?])", r" \1", s)
  s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
  return s


def read_parallel_corpus(lang1: str, lang2: str, reverse: bool = False) -> Tuple[LanguageDictionary,
                                                                                 LanguageDictionary,
                                                                                 List[List[str]]]:
  """Reads a parallel text corpus from a text file. Assumes that there are two columns separated by
  a tab character. Words are separated by spaces."""
  print("Reading lines...")

  # Read the file and split into lines
  txt_path = os.path.join('..', f'data/{lang1}-{lang2}.txt')
  lines = open(txt_path, encoding='utf-8').read().strip().split('\n')

  # Split every line into pairs and normalize
  pairs = [[normalize_string(s) for s in l.split('\t')] for l in lines]

  # Reverse pairs
  if reverse:
    pairs = [list(reversed(p)) for p in pairs]

  # Make language dictionaries
  return LanguageDictionary(lang1), LanguageDictionary(lang2), pairs


MAX_SENTENCE_LENGTH = 10

eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


def filter_pair(language_pairs: List[str]):
  assert len(language_pairs) == 2, "Should always be of length 2"
  return len(language_pairs[0].split(' ')) < MAX_SENTENCE_LENGTH and \
      len(language_pairs[1].split(' ')) < MAX_SENTENCE_LENGTH and \
      language_pairs[1].startswith(eng_prefixes)


def filter_pairs(pairs: List[List[str]]):
    return [pair for pair in pairs if filter_pair(pair)]


def prepare_data(lang1: str, lang2: str, reverse: bool = False):
  input_dict, output_dict, pairs = read_parallel_corpus(lang1, lang2, reverse)
  print(f"Read {len(pairs)} sentence pairs")
  pairs = filter_pairs(pairs)
  # breakpoint()
  print(f"Trimmed to {len(pairs)} sentence pairs")
  for text in pairs:
    input_dict.add_sentence(text[0])
    output_dict.add_sentence(text[1])
  print(f"Counted words:")
  print(f"Input language {lang1} - {input_dict.n_words}")
  print(f"Output language {lang2} - {output_dict.n_words}")
  return input_dict, output_dict, pairs


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    seconds = time.time() - since
    remaining_seconds = (seconds / percent) - seconds
    return '%s (- %s)' % (as_minutes(seconds), as_minutes(remaining_seconds))


def indexes_from_sentence(lang_dictionary: LanguageDictionary, sentence: str) -> List[int]:
  return [lang_dictionary.word2index[word] for word in sentence.split(' ')]


def tensor_from_sentence(lang_dictionary: LanguageDictionary, sentence: str) -> Tensor:
  indexes = indexes_from_sentence(lang_dictionary, sentence)
  indexes.append(LanguageDictionary.EOS_token)
  return torch.tensor(indexes, dtype=torch.long).view(-1, 1)


def tensors_from_pair(input_lang: LanguageDictionary, output_lang: LanguageDictionary,
                      pair: List[str]) -> Tuple[Tensor, Tensor]:
  input_tensor = tensor_from_sentence(input_lang, pair[0])
  target_tensor = tensor_from_sentence(output_lang, pair[1])
  return input_tensor, target_tensor


def write_losses(output_path: str, training_loss: List[float], validation_loss: List[float]):
  """Write training and validation losses to CSV file."""
  assert len(training_loss) == len(validation_loss), f"{len(training_loss)} " \
                                                     f"!= {len(validation_loss)}"
  with open(output_path, 'w') as f:
    writer = csv.writer(f)
    writer.writerow(['epoch', 'training_loss', 'validation_loss'])
    for i, (train, val) in enumerate(zip(training_loss, validation_loss)):
      writer.writerow([i, train, val])
  print(f"Train and validation losses are output to {output_path}")
