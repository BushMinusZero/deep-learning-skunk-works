from typing import Callable

import numpy as np
import torch
from torch.optim import Optimizer


class EarlyStopping:
    def __init__(self, patience: int = 7, verbose: bool = False, delta: int = 0,
                 model_path: str = 'checkpoint.pt', trace_func: Callable = print):
        """Early stops the training if validation loss doesn't improve after a given patience.
        :param patience (int): How long to wait after last time validation loss improved.
        :param verbose (bool): If True, prints a message for each validation loss improvement.
        :param delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        :param model_path (str): Path for the checkpoint to be saved to. Default: 'checkpoint.pt'
        :param trace_func (function): trace print function.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.model_path = model_path
        self.trace_func = trace_func

        self.attempt_counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def check_early_stopping_criteria(self, model: torch.nn.Module, optimizer: Optimizer,
                                      epoch: int, train_loss: float, val_loss: float) -> None:
        score = -val_loss
        if self.best_score is None:
          # No score history
          self.best_score = score
          self.save_model_checkpoint(model, optimizer, epoch, train_loss, val_loss)
          self.val_loss_min = val_loss
        elif score < self.best_score + self.delta:
          # Not a new best score
          self.attempt_counter += 1
          self.trace_func(f'EarlyStopping counter: {self.attempt_counter} out of {self.patience}')
          if self.attempt_counter >= self.patience:
            self.early_stop = True
        else:
          # Record a new best score
          self.best_score = score
          self.save_model_checkpoint(model, optimizer, epoch, train_loss, val_loss)
          self.val_loss_min = val_loss
          self.attempt_counter = 0

    def save_model_checkpoint(self, model: torch.nn.Module, optimizer: Optimizer,
                              epoch: int, train_loss: float, val_loss: float) -> None:
      """Saves model when validation loss decrease."""
      if self.verbose:
        self.trace_func(f'Validation loss decreased '
                        f'({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
      torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
      }, self.model_path)
