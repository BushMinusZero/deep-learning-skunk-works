import os
from datetime import datetime
from typing import Optional


class Config:
  pass


class CBOWConfig(Config):
  batch_size = 32
  embedding_dim = 64
  num_epochs = 10
  patience = 4
  learning_rate_decay = 0.9
  learning_rate_step_size = 1
  context_size = 4

  raw_data_path = os.path.join('data', 'eng.txt')
  output_dir = os.path.join('data', 'cbow')
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
    self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir, 'model.pt')
    self.model_vocab_path = os.path.join(self.model_checkpoint_dir, 'vocab.txt')


class SkipGramConfig(Config):
  batch_size = 32
  embedding_dim = 64
  num_epochs = 10
  patience = 4
  learning_rate_decay = 0.9
  learning_rate_step_size = 1

  raw_data_path = os.path.join('data', 'eng.txt')
  output_dir = os.path.join('data', 'skip_gram')
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
    self.model_checkpoint_path = os.path.join(self.model_checkpoint_dir, 'model.pt')
    self.model_vocab_path = os.path.join(self.model_checkpoint_dir, 'vocab.txt')
