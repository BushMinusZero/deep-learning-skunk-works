import os
from statistics import mean
from typing import Dict, List, Optional
from collections import defaultdict

from torch import Tensor

from src.utils import l2_norm, cosine_similarity
from src.word2vec import Word2VecConfig, InferenceServer, get_latest_word2vec_model


class Task:
  def __init__(self, a: str, b: str, c: str, d: str):
    """Analogy task requires data to measure relation a is to b as c is d."""
    self.a = a
    self.b = b
    self.c = c
    self.d = d

  def __repr__(self):
    return f'Task(a={self.a},b={self.b},c={self.c},d={self.d})'

  def get_words(self):
    return [self.a, self.b, self.c, self.d]


def load_analogy_labels(analogy_path: str) -> Dict[str, List[Task]]:
  """Load intrinsic evaluation - analogy task labels
  From paper 'Efficient Estimation of Word Representations in Vector Space'
  https://arxiv.org/pdf/1301.3781.pdf
  http://www.fit.vutbr.cz/~imikolov/rnnlm/word-test.v1.txt
  """
  with open(analogy_path) as f:
    category = None
    analogy_labels = defaultdict(list)
    for row in f:
      if row.startswith('//'):
        # ignore it
        pass
      elif row.startswith(':'):
        category = row.split(':')[-1].strip()
      else:
        words = row.split()
        assert len(words) == 4, "should be exactly 4 words in an analogy label"
        analogy_labels[category].append(Task(words[0], words[1], words[2], words[3]))
    return analogy_labels


if __name__ == '__main__':
  tasks_by_category = load_analogy_labels(os.path.join('data', 'word-test.v1.txt'))

  conf = Word2VecConfig(model_date=get_latest_word2vec_model())
  server = InferenceServer(conf)
  server.load_model(conf.model_checkpoint_path)
  for category, tasks in tasks_by_category.items():
    print(f'Category: {category}')
    scores = []
    scores_no_unknowns = []
    for task in tasks:
      embedded_words = [server.embed_word(w.lower()) for w in task.get_words()]
      score = cosine_similarity(embedded_words[0] - embedded_words[1],
                                embedded_words[2] - embedded_words[3]).item()
      if all([w in server.vocab for w in task.get_words()]):
        scores_no_unknowns.append(score)
      scores.append(score)
    num_unknowns = len(scores) - len(scores_no_unknowns)
    print(f'Found unknowns: {num_unknowns} out of {len(scores)}')
    print(f'Average score: {mean(scores)}')
    print(f'Average score (no unknowns): {mean(scores_no_unknowns) if scores_no_unknowns else None}')
