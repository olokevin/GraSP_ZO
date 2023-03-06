"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-10-24 16:27:47
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-10-24 16:27:47
"""
from typing import Callable

import numpy as np
import torch
from pyutils.general import logger
from pyutils.torch_train import (
    get_learning_rate,
    get_random_state,
    set_torch_deterministic,
    set_torch_stochastic,
)
from torch import nn
from torch.functional import Tensor
from torch.optim import Optimizer
from torchonn.op.mzi_op import RealUnitaryDecomposerBatch, checkerboard_to_vector, vector_to_checkerboard
from tensor_layers.layers import TensorizedLinear_linear, TensorizedLinear_module, TensorizedLinear_module_tonn
from tensor_fwd_bwd.tensorized_linear import TensorizedLinear

__all__ = ["ZO_SCD_mask"]


class ZO_SCD_mask(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        masks,
        lr: float = 0.1,
        grad_sparsity: float = 0.1,
        tensorized: str = 'None'
    ):
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self.forward_counter = 0
        self.global_step = 0
        self.model = model
        self.criterion = criterion
        self.masks = masks
        self.grad_sparsity = grad_sparsity
        self.tensorized = tensorized
        self.init_state()

    def init_state(self):
        self.modules = self.extract_modules(self.model)
        self.trainable_params = self.extract_trainable_parameters(self.model)
        # self.untrainable_params = self.extract_untrainable_parameters(self.model)

        if self.masks is not None:
            self.enable_mixedtraining(self.masks)
        else:
            self.disable_mixedtraining()


    def extract_modules(self, model):
        if self.tensorized == 'TensorizedLinear_module_tonn':
            return {
                layer_name: layer
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (TensorizedLinear_module_tonn))
            }
        if self.tensorized == 'TensorizedLinear_module':
            return {
                layer_name: layer
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (TensorizedLinear_module))
            }
        elif self.tensorized == 'TensorizedLinear':
            return {
                layer_name: layer
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (TensorizedLinear))
            }
        else:
            return {
                layer_name: layer
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (nn.Linear, nn.Conv2d))
            }

    def extract_trainable_parameters(self, model):
        # always flatten the parameters
        if self.tensorized == 'TensorizedLinear_module_tonn':
            return {
                layer_name: {
                    "tt_cores-"+str(i): getattr(layer,'tt_cores')[i].weight.view(-1)
                    for i in range(getattr(layer, 'order'))
                }
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (TensorizedLinear_module_tonn))
            }
        elif self.tensorized == 'TensorizedLinear_module':
            return {
                layer_name: {
                    str(i): getattr(layer.tensor,'factors')[i].view(-1)
                    for i in range(getattr(layer.tensor, 'order'))
                }
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (TensorizedLinear_module))
            }
        elif self.tensorized == 'TensorizedLinear':
            return {
                layer_name: {
                    str(i): getattr(layer.weight.factors, 'factor_'+str(i)).view(-1)
                    for i in range(getattr(layer.weight, 'order'))
                }
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (TensorizedLinear))
            }
        else:
            return {
                layer_name: {  
                    param_name: getattr(layer, param_name).view(-1)
                    for param_name in ["weight"]
                }
                for layer_name, layer in model.named_modules()
                if isinstance(layer, (nn.Linear, nn.Conv2d))
            }

    # def extract_untrainable_parameters(self, model):
    #     return {
    #         layer_name: {
    #             param_name: getattr(layer, param_name)
    #             for param_name in ["phase_bias_U", "phase_bias_V", "delta_list_U", "delta_list_V"]
    #         }
    #         for layer_name, layer in model.named_modules()
    #         if isinstance(layer, (MZIBlockLinear, MZIBlockConv2d))
    #     }

    def enable_mixedtraining(self, masks):
        # need to change to index [0, 1, 2, 3][0, 1, 0, 1] => [1, 3]
        self.mixedtrain_masks = {
            layer_name: {
                p_name: torch.arange(p.numel(), device=p.device)[
                    masks[layer_name][p_name].to(p.device).bool().view(-1)
                ]
                for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in self.trainable_params.items()
        }
        print(self.mixedtrain_masks)

    def disable_mixedtraining(self):
        self.mixedtrain_masks = {
            layer_name: {
                p_name: torch.arange(p.numel(), device=p.device) for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in self.trainable_params.items()
        }

    def commit(self, layer_name: str, param_name: str, param: Tensor) -> None:
        '''
            layer_name:{param_name: p}
        '''
        layer = self.modules[layer_name]
        if self.tensorized == 'TensorizedLinear_tonn':
            raise NotImplementedError
        if self.tensorized == 'TensorizedLinear_module':
            idx = int(param_name)
            ttm_shape = layer.tensor.factors[idx].shape
            layer.tensor.factors[idx].data = param.reshape(ttm_shape)
        elif self.tensorized == 'TensorizedLinear':
            idx = int(param_name)
            tt_shape = (layer.weight.rank[idx],
                        layer.weight.shape[idx],
                        layer.weight.rank[idx+1])
            # t_param = torch.nn.parameter.Parameter(param.reshape(tt_shape), requires_grad=False)
            # setattr(layer.weight.factors, 'factor_'+param_name, t_param)
            setattr(getattr(layer.weight.factors, 'factor_'+param_name), 'data', param.reshape(tt_shape))
        else:
            if param_name == "weight":
                layer.weight.data = param.reshape(layer.out_features, layer.in_features)
            else:
                raise ValueError(f"Wrong param_name {param_name}")

    def zo_coordinate_descent(self, obj_fn, params):
        """
        description: stochastic coordinate-wise descent.
        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)

        for layer_name, layer_params in params.items():
            layer_masks = self.mixedtrain_masks[layer_name]
            for p_name, p in layer_params.items():
                selected_indices = layer_masks[p_name]
                for idx in selected_indices:
                    # ============ SparseTune in FLOPS+ [Gu+, DAC 2020] ============
                    cur_seed = get_random_state()
                    set_torch_stochastic()
                    seed = np.random.rand()
                    set_torch_deterministic(cur_seed)
                    if seed < self.grad_sparsity:
                        continue
                    old_value = p.data[idx]
                    pos_perturbed_value = old_value + lr
                    neg_perturbed_value = old_value - lr

                    p.data[idx] = pos_perturbed_value
                    with torch.no_grad():  # training=True to enable profiling, but do not save graph
                        self.commit(layer_name, p_name, p)
                        y, new_loss = obj_fn()
                        self.forward_counter += 1

                    if new_loss < old_loss:
                        old_loss = new_loss
                    else:
                        p.data[idx] = neg_perturbed_value
                        with torch.no_grad():
                            self.commit(layer_name, p_name, p)
                            y, old_loss = obj_fn()
                            self.forward_counter += 1
        return y, old_loss

    def build_obj_fn(self, data, target, model, criterion):
        def _obj_fn():
            y = model(data)
            return y, criterion(y, target)

        return _obj_fn

    def step(self, data, target):
        self.obj_fn = self.build_obj_fn(data, target, self.model, self.criterion)
        y, loss = self.zo_coordinate_descent(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        return y, loss
