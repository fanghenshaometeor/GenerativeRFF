# Generative Random Fourier Features
Codes for the paper `End-to-end Kernel Learning via Generative Random Fourier Features` accepted by *Pattern Recognition*. **[journal](https://www.sciencedirect.com/science/article/pii/S0031320322005374)**, **[arxiv](https://arxiv.org/abs/2009.04614v4)**.

If our work is helpful for your research, please consider citing:
```
@article{fang2023end,
  title={End-to-end kernel learning via generative random Fourier features},
  author={Fang, Kun and Liu, Fanghui and Huang, Xiaolin and Yang, Jie},
  journal={Pattern Recognition},
  volume={134},
  pages={109057},
  year={2023},
  publisher={Elsevier}
}
```

## Table of Content
  - [1. File descriptions](#1file-descriptions)
  - [2. Train and attack](#2train-and-attack)

## 1. File descriptions

A brief description for the files in this repo:
- `model.py` definitions of the GRFF model
- `modelv.py` definitions of the variant of the GRFF model for image data
- `data_loader.py` scripts on loading the data
- `train.sh` & `train.py` scripts on training the GRFF model on *synthetic* data and real-world *benchmark* data
- `train_attack_mnist.sh` & `train_mnist.py` & `attack_mnist.py` scripts on training and attacking the GRFF variant on MNIST

## 2. Train and attack

### Generalization

To see the improved generalization performance of the GRFF model on the synthetic data and the real-world benchmark data, run
```
sh train.sh
```
Comment or uncomment specific lines in `train.sh` to run the corresponding experiments.

### Adversarial robustness
To see the adversarial robustness of the GRFF model on MNIST, run
```
sh train_attack_mnist.sh
```

Detailed settings of the training hyper-parameters can be found in the 2 scripts above.

##

If u have problems about the codes or paper, u could contact me (fanghenshao@sjtu.edu.cn) or raise issues in this repo.