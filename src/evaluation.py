import os
from statistics import mean
from typing import Dict, List
from collections import defaultdict

import pandas as pd

from src.config import CBOWConfig
from src.utils import cosine_similarity
from src.cbow import InferenceServer, get_latest_cbow_model


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


def evaluate_model_on_analogy_labels():
  tasks_by_category = load_analogy_labels(os.path.join('data', 'word-test.v1.txt'))
  model_date = get_latest_cbow_model()
  # TODO: make evaluation general to any pytorch model + vocab
  conf = CBOWConfig(model_date=model_date)
  server = InferenceServer(conf)
  server.load_model(conf.model_checkpoint_path)
  evaluation_stats = []
  overall_score, overall_score_no_unknowns, overall_count, overall_count_no_unknowns = 0, 0, 0, 0
  for category, tasks in tasks_by_category.items():
    scores, scores_no_unknowns = [], []
    for task in tasks:
      embedded_words = [server.embed_word(w.lower()) for w in task.get_words()]
      score = cosine_similarity(embedded_words[0] - embedded_words[1],
                                embedded_words[2] - embedded_words[3]).item()
      if all([w in server.vocab for w in task.get_words()]):
        scores_no_unknowns.append(score)
      scores.append(score)
    num_unknowns = len(scores) - len(scores_no_unknowns)
    overall_count += len(scores)
    overall_count_no_unknowns += num_unknowns
    overall_score += sum(scores)
    overall_score_no_unknowns += sum(scores_no_unknowns)
    evaluation_stats.append({
      'category': category,
      'average_score': mean(scores),
      'average_score_no_unknowns': mean(scores_no_unknowns) if scores_no_unknowns else None,
      'num_unknowns': num_unknowns,
      'model': model_date,
    })
  eval_df = pd.DataFrame(evaluation_stats)
  output_path = os.path.join(conf.model_checkpoint_dir, 'analogy_evaluation_stats.csv')
  eval_df.to_csv(output_path, index=False)
  summary_output_path = os.path.join(conf.model_checkpoint_dir, 'analogy_evaluation_summary_stats.csv')
  pd.DataFrame([
    {'name': 'average_score', 'count': overall_count, 'score': overall_score / overall_count},
    {'name': 'average_score_no_unknowns', 'count': overall_count_no_unknowns,
     'score': overall_score_no_unknowns / overall_count_no_unknowns},
  ]).to_csv(summary_output_path, index=False)


if __name__ == '__main__':
  evaluate_model_on_analogy_labels()
