"""
This file runs the main training/val loop, etc... using Lightning Trainer    
"""
import os
from pytorch_lightning import Trainer, seed_everything
from argparse import ArgumentParser
from src.production_mnist.mnist import CoolSystem
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
import torchvision.transforms as transforms

# sets seeds for numpy, torch, etc...
# must do for DDP to work well
seed_everything(123)

def main(args):
    # init module
    model = CoolSystem(hparams=args)

    train_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size)
    val_loader = DataLoader(MNIST(os.getcwd(), train=True, download=True, transform=transforms.ToTensor()), batch_size=args.batch_size)

    # makes all flags available to trainer from cli
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)


if __name__ == '__main__':
    parser = ArgumentParser(add_help=False)

    # add args from trainer
    parser = Trainer.add_argparse_args(parser)

    # give the module a chance to add own params
    # good practice to define LightningModule speficic params in the module
    parser = CoolSystem.add_model_specific_args(parser)

    # parse params
    args = parser.parse_args()

    main(args)
