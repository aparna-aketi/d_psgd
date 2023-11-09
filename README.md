# D-PSGD
This repository implements Decentralized Parallel Stochastic Gradient Descent ([D-PSGD](https://arxiv.org/abs/1705.09056)) algorithm.

# Available Models
* MobileNet-V2
* ResNet
* VGG11

# Requirements
* found in env.yml file

# Hyper-parameters
* --world_size   = total number of agents
* --graph        = graph topology (default ring)
* --neighbors    = number of neighbor per agent (default 2)
* --arch         = model to train
* --normtype     = type of normalization layer
* --dataset      = dataset to train
* --batch_size   = batch size for training
* --epochs       = total number of training epochs
* --lr           = learning rate
* --momentum     = momentum coefficient
* --nesterov     = activates nesterov momentum
* --gamma        = averaging rate for gossip 
* --alpha        = amount of skew in the data distribution (alpha of Dirichlet distribution); 0.01 = completely non-iid and 10 = more towards iid
* --weight_decay = L2 regularization coefficient

# How to run?

test file shows sample commands to run the code
```
sh test.sh
```


