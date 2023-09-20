import logging
import torch
import pytorch_lightning as pl
from pytorch_lightning.cli import LightningCLI

pl_logger = logging.getLogger('pytorch_lightning')

if __name__ == '__main__':
	import datetime
	pl_logger.info(f"Starting at {datetime.datetime.now()}")

	torch.set_float32_matmul_precision('medium')

	cli = LightningCLI(
		trainer_class=pl.Trainer, 
		save_config_callback=None,
	)
