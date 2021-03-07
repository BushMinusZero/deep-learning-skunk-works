import os
from datetime import datetime
from typing import Optional


class Config:
  batch_size: int = NotImplemented
  embedding_dim: int = NotImplemented
  num_epochs: int = NotImplemented
  patience: int = NotImplemented
  learning_rate_decay: float = NotImplemented
  learning_rate_step_size: int = NotImplemented
  context_size: int = NotImplemented

  raw_data_path: str = NotImplemented
  output_dir: str = NotImplemented
  data_dir: str = NotImplemented
  features_dir: str = NotImplemented
  model_root_dir: str = NotImplemented

  model_checkpoint_dir: str = NotADirectoryError
  model_best_checkpoint_path: str = NotImplemented
  model_latest_checkpoint_path: str = NotImplemented
  model_vocab_path: str = NotImplemented

  @classmethod
  def get_latest_model(cls, date_fmt: str = '%Y-%m-%dT%H:%M:%S') -> str:
    model_dirs = os.listdir(cls.model_root_dir)
    dates = [datetime.strptime(d, date_fmt) for d in model_dirs]
    return max(dates).strftime(date_fmt)

  @classmethod
  def get_project_root(cls):
    project_root = os.getenv('PROJECT_ROOT')
    assert project_root, "Env variable PROJECT_ROOT not set."
    assert os.path.exists(project_root), f"Path PROJECT_ROOT={project_root} does not exist."
    return project_root


class CBOWConfig(Config):
  batch_size = 32
  embedding_dim = 64
  num_epochs = 10
  patience = 4
  learning_rate_decay = 0.9
  learning_rate_step_size = 1
  context_size = 4

  project_root = Config.get_project_root()
  raw_data_path = os.path.join(project_root, 'data', 'eng.txt')
  output_dir = os.path.join(project_root, 'data', 'cbow')
  data_dir = os.path.join(output_dir, 'data')
  features_dir = os.path.join(output_dir, 'features')
  model_root_dir = os.path.join(output_dir, 'models')

  def __init__(self, model_date: Optional[str] = None):
    now = (model_date if model_date else datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
    self.model_checkpoint_dir = os.path.join(self.model_root_dir, now)
    os.makedirs(self.data_dir, exist_ok=True)
    os.makedirs(self.features_dir, exist_ok=True)
    os.makedirs(self.data_dir, exist_ok=True)
    os.makedirs(self.model_root_dir, exist_ok=True)
    os.makedirs(self.model_checkpoint_dir, exist_ok=True)
    self.model_best_checkpoint_path = os.path.join(self.model_checkpoint_dir, 'model_best.pt')
    self.model_latest_checkpoint_path = os.path.join(self.model_checkpoint_dir, 'model_latest.pt')
    self.model_vocab_path = os.path.join(self.model_checkpoint_dir, 'vocab.txt')


class SkipGramConfig(Config):
  batch_size = 32
  embedding_dim = 64
  num_epochs = 20
  patience = 5
  learning_rate_decay = 0.9
  learning_rate_step_size = 1
  context_size = 4

  project_root = Config.get_project_root()
  raw_data_path = os.path.join(project_root, 'data', 'eng.txt')
  output_dir = os.path.join(project_root, 'data', 'skip_gram')
  data_dir = os.path.join(output_dir, 'data')
  features_dir = os.path.join(output_dir, 'features')
  model_root_dir = os.path.join(output_dir, 'models')

  def __init__(self, model_date: Optional[str] = None):
    now = (model_date if model_date else datetime.now().strftime('%Y-%m-%dT%H:%M:%S'))
    self.model_checkpoint_dir = os.path.join(self.model_root_dir, now)
    os.makedirs(self.data_dir, exist_ok=True)
    os.makedirs(self.features_dir, exist_ok=True)
    os.makedirs(self.data_dir, exist_ok=True)
    os.makedirs(self.model_root_dir, exist_ok=True)
    os.makedirs(self.model_checkpoint_dir, exist_ok=True)
    self.model_best_checkpoint_path = os.path.join(self.model_checkpoint_dir, 'model_best.pt')
    self.model_latest_checkpoint_path = os.path.join(self.model_checkpoint_dir, 'model_latest.pt')
    self.model_vocab_path = os.path.join(self.model_checkpoint_dir, 'vocab.txt')

