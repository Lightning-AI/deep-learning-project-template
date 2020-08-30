## Research Seed Folder   
Create a folder for each contribution (ie: MNIST, BERT, etc...).   
Each folder will have:

##### contribution_name_trainer.py    
Runs your LightningModule. Abstracts training loop, distributed training, etc...   

##### contribution_name.py  
Holds your main contribution   

## Example  
The folder here gives an example for mnist.   

### MNIST    
In this readme, give instructions on how to run your code.   

#### CPU   
```bash   
python mnist_trainer.py     
```

#### Multiple-GPUs   
```bash   
python mnist_trainer.py --gpus 4
```   

or specific GPUs
```bash   
python mnist_trainer.py --gpus '0,3'
```   

#### On multiple nodes   
```bash  
python mnist_trainer.py --gpus 4 --nodes 4  --precision 16
```   
