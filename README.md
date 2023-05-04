# Zeroth-order Training for Lottery-pruned Models & Tensor-compressed Models

This is a PyTorch implementation of Zeroth-order training for MNIST dataset.

* Sparse training: Prune at initialization (GraSP https://arxiv.org/abs/2002.07376)
* Tensor-compressed models: TT-format and TTM-format

# Usage:

## MNIST

For GraSP-pruned FC layers:

```bash
# FO-benchmark
python -u main_prune_MNIST.py -config configs/MNIST/FC/FO.yml

# ZO-gradient estimator
python main_prune_MNIST.py -config configs/MNIST/TTM/SGD.yml

# ZO-finite difference
python -u main_prune_MNIST.py -config configs/MNIST/FC/SCD_esti.yml
# ZO-coordinate descent
python -u main_prune_MNIST.py -config configs/MNIST/FC/SCD_batch.yml
```

For TTM layers:

```bash
# FO-benchmark
python -u main_prune_MNIST.py -config configs/MNIST/TTM/FO.yml

# ZO-gradient estimator
python main_prune_MNIST.py -config configs/MNIST/TTM/SGD.yml

# ZO-finite difference
python main_prune_MNIST.py -config configs/MNIST/TTM/SCD_esti.yml
# ZO-coordinate descent
python main_prune_MNIST.py -config configs/MNIST/TTM/SCD_batch.yml
```



## 2-layer Encoder:

* Select the provided experiments in ./run_tensors.sh
* run:

```
./run_tensors.sh
```



# Zeroth-order Optimizer

## ZO_SGD_mask:

./optimizer/ZO_SGD_mask.py

Based on stochastic gradient estimator

* perturb all parameters with i.i.d. Gaussian perturbation
* evaluate the change of Loss function -> evaluate the directional direction of selected random direction
* get single-shot gradient estimation
* The expectation of gradient estimation is a bounded bias estimation of the true gradient 

<img src="C:\Users\KevinZ\AppData\Roaming\Typora\typora-user-images\image-20230504114916795.png" alt="image-20230504114916795" style="zoom:50%;" />

```python
def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        masks,
        lr: float = 0.01,
        sigma: float = 0.1,
        n_sample: int = 20,
        signSGD: bool = False,
        layer_by_layer: bool = False,
        opt_layers_strs: list = []
    ):
```

Related Work:

[FLOPS: EFficient On-Chip Learning for OPtical Neural Networks Through Stochastic Zeroth-Order Optimization | IEEE Conference Publication | IEEE Xplore](https://ieeexplore.ieee.org/document/9218593)



## ZO_SCD_mask

./optimizer/ZO_SGD_mask.py

```python
def __init__(
        self,
        model: nn.Module,	# 
        criterion: Callable,
        masks,
        lr: float = 0.1,
        grad_sparsity: float = 0.1,
        h_smooth: float = 0.001,
        grad_estimator: str = 'sign',
        opt_layers_strs: list = [],
        STP: bool = True,
        momentum: float = 0,
        weight_decay: float = 0,
        dampening: float = 0,
        adam: bool = False,
        beta_1: float = 0.9,
        beta_2: float = 0.98,
        eps: float = 1e-06
    ):
```

<img src="C:\Users\KevinZ\AppData\Roaming\Typora\typora-user-images\image-20230504115526548.png" alt="image-20230504115526548" style="zoom:33%;" />

<img src="C:\Users\KevinZ\AppData\Roaming\Typora\typora-user-images\image-20230504115642767.png" alt="image-20230504115642767" style="zoom:50%;" />

grad_estimator: update rule

* 'sign': ZO-det Coordinate Descent, update the parameter one-by-one

* 'batch': ZO-det Coordinate Descent, update all parameters at the end of evaluation

* 'esti': ZO-finite difference, update all parameters at the end of evaluation

opt_layers_strs: layers that need to be trained. now supports:

* 'nn.Linear': nn.Linear,
* 'nn.Conv2d': nn.Conv2d,
* 'TensorizedLinear': TensorizedLinear,
* 'TensorizedLinear_module': TensorizedLinear_module,
* 'TensorizedLinear_module_tonn': TensorizedLinear_module_tonn

Related Work:

https://ojs.aaai.org/index.php/AAAI/article/view/16928
