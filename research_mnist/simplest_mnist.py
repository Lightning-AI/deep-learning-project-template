"""
This file defines the core research contribution.
"""
from argparse import ArgumentParser

import pytorch_lightning as pl
import torch
from torch.nn import functional as F


class CoolSystem(pl.LightningModule):

    def __init__(self, hparams):
        super(CoolSystem, self).__init__()
        # not the best model...
        self.hparams = hparams
        self.l1 = torch.nn.Linear(28 * 28, 10)

    def forward(self, x):
        return torch.relu(self.l1(x.view(x.size(0), -1)))

    def training_step(self, batch, batch_idx):
        # REQUIRED
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)

        result = pl.TrainResult(minimize=loss)
        result.log('train_loss', loss, prog_bar=True)

        return result

    def validation_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult()
        result.valid_batch_loss = loss
        result.log('valid_loss', loss, on_epoch=True, prog_bar=True)

        return result

    def validation_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = outputs.valid_batch_loss.mean()
        result = pl.EvalResult(checkpoint_on=avg_loss)
        result.log('valid_loss', avg_loss, on_epoch=True, prog_bar=True)

        return result

    def test_step(self, batch, batch_idx):
        # OPTIONAL
        x, y = batch
        y_hat = self.forward(x)
        loss = F.cross_entropy(y_hat, y)
        result = pl.EvalResult()
        result.test_batch_loss = loss
        result.log('test_loss', loss, on_epoch=True)

        return result

    def test_epoch_end(self, outputs):
        # OPTIONAL
        avg_loss = outputs.test_batch_loss.mean()

        result = pl.EvalResult()
        result.log('test_loss', avg_loss, on_epoch=True)
        return result

    def configure_optimizers(self):
        # REQUIRED
        # can return multiple optimizers and learning_rate schedulers
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--learning_rate', default=0.02, type=float)

        # training specific (for this model)
        parser.add_argument('--max_nb_epochs', default=2, type=int)

        return parser
