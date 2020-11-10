from pytorch_lightning import Trainer, seed_everything
from torch.utils import data
from project.lit_mnist import LitClassifier
from project.lit_mnist import LitMNISTDataModule


def test_lit_classifier():
    seed_everything(1234)

    model = LitClassifier()
    mnist = LitMNISTDataModule(data_dir="", batch_size=32)

    trainer = Trainer(limit_train_batches=50, limit_val_batches=20, max_epochs=2)
    trainer.fit(model, mnist)

    results = trainer.test(datamodule=mnist)
    assert results[0]['test_acc'] > 0.7
