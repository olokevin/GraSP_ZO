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

__all__ = ["ZO_SCD_grad"]


class ZO_SCD_grad(Optimizer):
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
        
        # if self.patience_table == True:
        #     self.enable_patience_table()


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
        # need to change to index 
        # matrix: [4, 5, 6, 7]
        # mask:   [0, 1, 0, 1]
        #  =>     [1, 3]
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

    # ============ apply gradients all together ============
    def _apply_gradients(self, params, grad, lr):
        return {
            layer_name: {p_name: p.add_(grad[layer_name][p_name]) for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }
    
    def zo_coordinate_descent(self, obj_fn, params):
        """
        description: stochastic coordinate-wise descent.
        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)
        grads = dict()

        for layer_name, layer_params in params.items():
            layer_masks = self.mixedtrain_masks[layer_name]
            layer_grads = dict()

            for p_name, p in layer_params.items():
                selected_indices = layer_masks[p_name]
                # param_grad: same size of p, masked remained 0
                param_grad = torch.zeros_like(p)

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

                    with torch.no_grad():  # training=True to enable profiling, but do not save graph
                        p.data[idx] = pos_perturbed_value
                        self.commit(layer_name, p_name, p)
                        y, pos_loss = obj_fn()
                        self.forward_counter += 1

                        p.data[idx] = neg_perturbed_value
                        self.commit(layer_name, p_name, p)
                        y, neg_loss = obj_fn()
                        self.forward_counter += 1

                        # loss_list = [old_loss, pos_loss, neg_loss]
                        # grad_list = [0, lr, -lr]

                        loss_list = [old_loss, pos_loss]
                        grad_list = [-lr, lr]

                        param_grad[idx] = grad_list[loss_list.index(min(loss_list))]

                        p.data[idx] = old_value
                        self.commit(layer_name, p_name, p)

                layer_grads[p_name] = param_grad

            grads[layer_name] = layer_grads

        # different from others!
        self._apply_gradients(params, grads, lr)
        return y, old_loss

    def build_obj_fn(self, data, target, model, criterion):
        def _obj_fn():
            y = model(data)
            return y, criterion(y, target)

        return _obj_fn

    def build_obj_fn_ATIS(self, datas, targets, model, criterion):
        def _obj_fn():
            # optimizer.step((w1,attn,seg),(target,slot_label))
            w1 = datas[0]
            attn = datas[1]
            seg = datas[2]

            target = targets[0]
            slot_label = targets[1]

            pred,pred_slot = model(w1,attn=attn,seg=seg)

            pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
            slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)

            loss_MLM =  criterion(pred_slot, slot_label)
            loss = criterion(pred,target)  + loss_MLM
            
            return (pred, pred_slot), loss
        return _obj_fn
    
    def step(self, data, target, ATIS=False):
        if ATIS == True:
            self.obj_fn = self.build_obj_fn_ATIS(data, target, self.model, self.criterion)
        else:
            self.obj_fn = self.build_obj_fn(data, target, self.model, self.criterion)
            
        y, loss = self.zo_coordinate_descent(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        return y, loss
