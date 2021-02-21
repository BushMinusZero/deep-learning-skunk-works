import argparse
import cbow
import evaluation


def parse_args():
  parser = argparse.ArgumentParser()
  # TODO: add model type arguments: cbow, skipgram, glove, ...
  # parser.add_argument('-m', '--model', action='store_true', help='Specify a model type')
  parser.add_argument('-t', '--train', action='store_true', help='Train a model')
  parser.add_argument('-i', '--infer', action='store_true', help='Run model inference')
  parser.add_argument('-e', '--eval', action='store_true', help='Run model evaluation')
  return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    if args.train:
      cbow.train()
    if args.infer:
      cbow.inference()
    if args.eval:
      evaluation.evaluate_model_on_analogy_labels()
