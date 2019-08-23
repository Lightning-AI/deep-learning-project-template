### First use    
Clone the repo    

```python
git clone https://github.com/williamFalcon/pytorch-lightning-conference-seed    
```   

Install the package so you can access everything use package references  
```python
cd pytorch-lightning-conference-seed   
pip install -e .   

# now you can do:
from research_seed import Whatever   
```    

### Running LightningModules    
A [LightningModule](https://pytorch-lightning.readthedocs.io/en/latest/LightningModule/RequiredTrainerInterface/) has the core logic to your research code. This includes:   
- Dataloaders (train, test, val)   
- Optimizers
- Training loop actions  
- Validation loop actions  

To run the module, feed it to the [Trainer](https://pytorch-lightning.readthedocs.io/en/latest/Trainer/) which handles mixed precision, checkpointing, multi-node, multi-GPU, etc...   
This makes your research contribution very clear and abstracted from the engineering details.    

To run the MNIST example in this package use the following command   
```python
# On CPU
python lightning_modules/train.py   

# On multiple GPUs (4 gpus here)
python lightning_modules/train.py --gpus '0,1,2,3'    

# On multiple nodes (16 gpus here)    
python lightning_modules/train.py --gpus '0,1,2,3' --nodes 4   
```   

### How to use this seed for research   
For each project define a new LightningModule. 

```python
import pytorch_lightning as pl   

class CoolerNotBERT(pl.LightningModule):
    def __init__(self):
        self.net = ...

    def training_step(self, batch, batch_nb):
        # do some other cool task
        # return loss   
```   

If you have variations of the same project it makes sense to use the same Module
```python
class BERT(pl.LightningModule):
    def __init__(self, model_name, task):
        self.task = task

        if model_name == 'transformer':
            self.net = Transformer()
        elif model_name == 'my_cool_version':
            self.net = MyCoolVersion()

    def training_step(self, batch, batch_nb):
        if self.task == 'standard_bert':
            # do standard bert training with self.net...
            # return loss

        if self.task == 'my_cool_task':
            # do my own version with self.net
            # return loss
```   

Then decide which to run using the trainer.   
```python
if use_bert:
    model = BERT()
else:
    model = CoolerNotBERT()

trainer = Trainer(gpus=[0, 1, 2, 3], use_amp=True)
trainer.fit(model)
```

### Trainer   
It's recommended that you have a single trainer per lightning module. However, you can also use a single trainer for all your LightningModules.    

Check out the [MNIST example](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/mnist).  
