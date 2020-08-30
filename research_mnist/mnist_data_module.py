from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
# Note - you must have torchvision installed for this example
from torchvision.datasets import MNIST


class MNISTDataModule(pl.LightningDataModule):

    def __init__(self, hparams):

        super().__init__()

        self.hparams = hparams

        self.data_dir = self.hparams.data_dir
        self.batch_size = self.hparams.batch_size

        # We hardcode dataset specific stuff here.
        self.num_classes = 10
        self.dims = (1, 28, 28)
        self.transform = transforms.Compose([transforms.ToTensor(), ])

        # Basic test that parameters passed are sensible.
        assert (
            self.hparams.train_size + self.hparams.valid_size == 60_000
        ), "Invalid Train and Valid Split, make sure they add up to 60,000"

    def prepare_data(self):
        # download the dataset
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):

        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform)
            self.mnist_train, self.mnist_val = random_split(
                mnist_full, [self.hparams.train_size, self.hparams.valid_size]
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            self.mnist_test = MNIST(
                self.data_dir, train=False, transform=self.transform
            )

    def train_dataloader(self):
        # REQUIRED
        return DataLoader(
            self.mnist_train,
            batch_size=self.batch_size,
            num_workers=self.hparams.workers,
        )

    def val_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.mnist_val, batch_size=self.batch_size, num_workers=self.hparams.workers
        )

    def test_dataloader(self):
        # OPTIONAL
        return DataLoader(
            self.mnist_test,
            batch_size=self.batch_size,
            num_workers=self.hparams.workers,
        )

    @staticmethod
    def add_data_specific_args(parent_parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # Dataset specific
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--batch_size", default=32, type=int)
        parser.add_argument("--data_dir", default="./", type=str)

        # training specific
        parser.add_argument("--train_size", default=55_000, type=int)
        parser.add_argument("--valid_size", default=5_000, type=int)
        parser.add_argument("--workers", default=8, type=int)

        return parser
