### pytorch-lightning-conference-seed
Use this seed to refactor your PyTorch research code for:  
- a paper submission  
- a new research project.     

[Read the usage instructions here](https://github.com/williamFalcon/pytorch-lightning-conference-seed/blob/master/HOWTO.md)

#### Goals  
The goal of this seed is to structure ML paper-code the same so that work can easily be extended and replicated.   

###### DELETE EVERYTHING ABOVE FOR YOUR PROJECT   
---   
<div align="center">    
 
# Your Project Name     

[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/NeurIPS-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/ICLR-2019-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-31-2018)  
<!--
ARXIV   
[![Paper](http://img.shields.io/badge/arxiv-math.co:1480.1111-B31B1B.svg)](https://www.nature.com/articles/nature14539)
-->



<!--  
Conference   
-->   
</div>
 
## Description   
What it does   

## How to run   
First, install dependencies   
```bash
# clone project   
git clone https://github.com/YourGithubName/Your-project-name   

# install project   
cd Your-project-name 
pip install -e .   
pip install requirements.txt
 ```   
 Next, navigate to [Your Main Contribution (MNIST here)] and run it.   
 ```bash
# module folder
cd research_seed/mnist/   

# run module (example: mnist as your main contribution)   
python mnist_trainer.py    
```

## Main Contribution      
List your modules here. Each module contains all code for a full system including how to run instructions.   
- [MNIST](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/mnist)  

## Baselines    
List your baselines here.   
- [MNIST_baseline](https://github.com/williamFalcon/pytorch-lightning-conference-seed/tree/master/research_seed/baselines/mnist_baseline)  

### Citation   
```
@article{YourName,
  title={Your Title},
  author={Your team},
  journal={Location},
  year={Year}
}
```   
