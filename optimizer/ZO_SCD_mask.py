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

__all__ = ["ZO_SCD_mask"]


class ZO_SCD_mask(Optimizer):
    def __init__(
        self,
        model: nn.Module,
        criterion: Callable,
        masks,
        lr: float = 0.1,
        grad_sparsity: float = 0.1
    ):
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self.forward_counter = 0
        self.global_step = 0
        self.model = model
        self.criterion = criterion
        self.masks = masks
        self.init_state()

    def init_state(self):
        self.modules = self.extract_modules(self.model)
        self.trainable_params = self.extract_trainable_parameters(self.model)
        # self.untrainable_params = self.extract_untrainable_parameters(self.model)
            
        self.enable_mixedtraining(self.masks)


    def extract_modules(self, model):
        return {
            layer_name: layer
            for layer_name, layer in model.named_modules()
            if isinstance(layer, (nn.Linear, nn.Conv2d))
        }

    def extract_trainable_parameters(self, model):
        # always flatten the parameters
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

    def commit(self, layer_name: str, param_name: str, phases: Tensor) -> None:
        layer = self.modules[layer_name]
        if param_name == "phase_U":
            phase_bias = self.untrainable_params[layer_name]["phase_bias_U"]
            delta_list = self.untrainable_params[layer_name]["delta_list_U"]
            quantizer = self.quantizers[layer_name]["phase_U_quantizer"]
            layer.U.data.copy_(
                self.decomposer.reconstruct(
                    delta_list,
                    self.v2m(quantizer(phases.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
                )
            )
        elif param_name == "phase_V":
            phase_bias = self.untrainable_params[layer_name]["phase_bias_V"]
            delta_list = self.untrainable_params[layer_name]["delta_list_V"]
            quantizer = self.quantizers[layer_name]["phase_V_quantizer"]

            layer.V.data.copy_(
                self.decomposer.reconstruct(
                    delta_list,
                    self.v2m(quantizer(phases.view(phase_bias.size(0), phase_bias.size(1), -1)) + phase_bias),
                )
            )

        elif param_name == "phase_S":
            layer.S.data.copy_(phases.data.cos().view_as(layer.S).mul_(layer.S_scale))
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
