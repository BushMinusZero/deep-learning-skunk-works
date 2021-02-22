from src.config import SkipGramConfig
from src.models.skipgram import SkipGramModel


if __name__ == '__main__':
  config = SkipGramConfig()
  vocab_size = 0  # TODO: compute vocab size
  model = SkipGramModel(vocab_size, config.embedding_dim)
