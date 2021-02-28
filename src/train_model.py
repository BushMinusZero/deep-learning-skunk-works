from typing import Generator, List, Union, Any

import torch
from torch import nn, optim, Tensor
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchtext.data import BucketIterator
from tqdm import tqdm

from src.config import Config
from src.early_stopping import EarlyStopping


class TrainingLoop:

  def __init__(self, config: Config, model: nn.Module, optimizer: optim.Optimizer,
               train_iterator: DataLoader, val_iterator: DataLoader):
    self.config = config
    self.model = model
    self.train_iterator = train_iterator
    self.val_iterator = val_iterator
    self.optimizer = optimizer

    # Learning rate scheduler
    self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.config.learning_rate_step_size,
                                               gamma=self.config.learning_rate_decay)

    # Initialize loss function
    self.train_losses = []
    self.val_losses = []
    self.test_losses = []

    self.num_epochs = self.config.num_epochs

    # Initialize early stopping criteria
    self.early_stopping = EarlyStopping(
      model_path=self.config.model_checkpoint_path,
      patience=self.config.patience
    )

    # update learning rate
    print(f"Learning rate: {self.scheduler.get_lr()[0]}")
    self.scheduler.step()

    # initialize tensorboard writer
    self.writer = SummaryWriter(self.config.model_checkpoint_dir)

  def save_model_checkpoint(self, epoch: int):
    torch.save({
      'epoch': epoch,
      'model_state_dict': self.model.state_dict(),
      'optimizer_state_dict': self.optimizer.state_dict(),
      'train_loss': self.train_losses[-1],
      'val_loss': self.val_losses[-1],
    }, self.config.model_checkpoint_path)

  def iterate_epoch(self) -> Generator[int, None, None]:

    for epoch in range(self.num_epochs):
      print(f'Starting epoch {epoch}')
      # Iterate over the epochs and train model here
      yield epoch

      print(f'Train loss: {self.train_losses[-1]}')
      print(f'Val loss:   {self.val_losses[-1]}')
      self.early_stopping.check_early_stopping_criteria(self.model, self.optimizer, epoch,
                                                        train_loss=self.train_losses[-1],
                                                        val_loss=self.val_losses[-1])
      if self.early_stopping.early_stop:
        # End epoch iteration if we meet the early stopping criteria
        print(f"Triggered early stopping criteria. Stopping after the {epoch}th iteration")
        break

  def compute_loss_from_batch(self, batch: Any) -> Tensor:
    raise NotImplementedError("Should be implemented by parent class.")

  def iterate_train(self, epoch: int) -> None:
    epoch_train_loss = 0
    self.model.train()
    for batch in tqdm(self.train_iterator):
      # batch is of type torchtext.data.batch.Batch
      # batch has dimension X x N where X is the max sequence length and N is the batch_size
      # the values in batch are the indices of the word in data_manager.source_vocab.itos

      # Before passing in a new instance, you need to zero out the gradients from the old instance
      self.model.zero_grad()

      loss = self.compute_loss_from_batch(batch)

      # Do the backward pass and update the gradient
      loss.backward()
      self.optimizer.step()

      # Update the total loss for the current epoch
      epoch_train_loss += loss.item()
      self.writer.add_scalar('Loss/train', epoch_train_loss, epoch)
      self.writer.add_scalar('Avg Loss/train', epoch_train_loss / len(self.train_iterator), epoch)

    # Record this epochs total loss
    self.train_losses.append(epoch_train_loss)

  def compute_loss(self, data_iterator: Union[BucketIterator, DataLoader],
                   losses: List[int]) -> float:
    self.model.eval()
    epoch_loss = 0
    for batch in tqdm(data_iterator):
      # Do a forward pass and compute the loss
      loss = self.compute_loss_from_batch(batch)
      epoch_loss += loss.item()
    losses.append(epoch_loss)
    return epoch_loss

  def compute_validation_loss(self, epoch: int) -> None:
    loss = self.compute_loss(self.val_iterator, self.val_losses)
    self.writer.add_scalar('Loss/val', loss, epoch)
    self.writer.add_scalar('Avg Loss/val', loss / len(self.val_iterator), epoch)

  def train(self):
    for epoch in self.iterate_epoch():
      # Forward and backward pass on training data
      self.iterate_train(epoch)

      # Forward pass on validation data
      self.compute_validation_loss(epoch)
