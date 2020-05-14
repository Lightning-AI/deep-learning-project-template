## MNIST    
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
