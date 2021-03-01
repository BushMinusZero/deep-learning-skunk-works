import os
from statistics import mean
from typing import Dict, List, Callable
from collections import defaultdict

import pandas as pd

from src.config import CBOWConfig, SkipGramConfig
from src.skipgram import load_skip_gram_model, load_counter_stoi_and_itos
from src.utils import cosine_similarity
from src.cbow import InferenceServer, DataManager, load_cbow_model


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


def evaluate_model_on_analogy_labels(embed_model_fn: Callable, model_dir: str,
                                     vocab: Dict[str, int]):
  tasks_by_category = load_analogy_labels(os.path.join('data', 'word-test.v1.txt'))
  evaluation_stats = []
  overall_score, overall_score_no_unknowns, overall_count, overall_count_no_unknowns = 0, 0, 0, 0
  for category, tasks in tasks_by_category.items():
    scores, scores_no_unknowns = [], []
    for task in tasks:
      embedded_words = [embed_model_fn(w) for w in task.get_words()]
      score = cosine_similarity(embedded_words[0] - embedded_words[1],
                                embedded_words[2] - embedded_words[3]).item()
      if all([w in vocab for w in task.get_words()]):
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
      'model': model_dir,
    })
  eval_df = pd.DataFrame(evaluation_stats)
  output_path = os.path.join(model_dir, 'analogy_evaluation_stats.csv')
  eval_df.to_csv(output_path, index=False)
  summary_output_path = os.path.join(model_dir,
                                     'analogy_evaluation_summary_stats.csv')
  pd.DataFrame([
    {'name': 'average_score', 'count': overall_count, 'score': overall_score / overall_count},
    {'name': 'average_score_no_unknowns', 'count': overall_count_no_unknowns,
     'score': overall_score_no_unknowns / overall_count_no_unknowns},
  ]).to_csv(summary_output_path, index=False)


def evaluate_cbow_model_on_analogy_labels():
  config = CBOWConfig(model_date=CBOWConfig.get_latest_model())
  data_manager = DataManager(config.batch_size)
  vocab = data_manager.read_vocab(config.model_vocab_path)
  model = load_cbow_model(config.model_best_checkpoint_path, vocab, config)
  server = InferenceServer(config, model, vocab)

  evaluate_model_on_analogy_labels(
    lambda x: server.embed_word(x.lower()),
    config.model_checkpoint_dir,
    server.vocab
  )


def evaluate_skip_gram_model_on_analogy_labels():
  config = SkipGramConfig(model_date=SkipGramConfig.get_latest_model())
  _, stoi, _ = load_counter_stoi_and_itos(config.model_vocab_path)
  model = load_skip_gram_model(config.model_best_checkpoint_path, stoi, config)
  server = InferenceServer(config, model, stoi)

  evaluate_model_on_analogy_labels(
    lambda x: server.embed_word(x.lower()),
    config.model_checkpoint_dir,
    server.vocab
  )


if __name__ == '__main__':
  evaluate_cbow_model_on_analogy_labels()
