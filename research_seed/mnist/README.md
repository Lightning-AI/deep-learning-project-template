## MNIST    
In this readme, give instructions on how to run your code.   

#### CPU   
```python   
python lightning_modules/train.py     
```

#### Multiple-GPUs   
```python   
python lightning_modules/train.py --gpus '0,1,2,3'  
```   

#### On multiple nodes   
```python  
python lightning_modules/train.py --gpus '0,1,2,3' --nodes 4  
```   
