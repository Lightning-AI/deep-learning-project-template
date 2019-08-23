## MNIST    
In this readme, give instructions on how to run your code.   

#### CPU   
```python   
python mnist_train.py     
```

#### Multiple-GPUs   
```python   
python mnist_train.py --gpus '0,1,2,3'  
```   

#### On multiple nodes   
```python  
python mnist_train.py --gpus '0,1,2,3' --nodes 4  
```   
