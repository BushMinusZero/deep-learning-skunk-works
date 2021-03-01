import argparse

import cbow
import evaluation
import skipgram


def parse_args():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', type=str, help='Specify a model type',
                      choices=['cbow', 'skipgram', 'glove'])
  parser.add_argument('-t', '--train', action='store_true', help='Train a model')
  parser.add_argument('-i', '--infer', action='store_true', help='Run model inference')
  parser.add_argument('-e', '--eval', action='store_true', help='Run model evaluation')
  return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.model == 'cbow':
      if args.train:
        cbow.train()
      if args.infer:
        cbow.inference()
      if args.eval:
        evaluation.evaluate_cbow_model_on_analogy_labels()
    elif args.model == 'skipgram':
      if args.train:
        skipgram.train()
      if args.eval:
        evaluation.evaluate_skip_gram_model_on_analogy_labels()
    elif args.model == 'glove':
      raise NotImplementedError
