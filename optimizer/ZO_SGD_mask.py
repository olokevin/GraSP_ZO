from typing import Callable

import copy
import numpy as np
import torch
from pyutils.general import logger
from pyutils.torch_train import get_learning_rate
from torch import nn
from torch.functional import Tensor
from torch.optim import Optimizer

from tensor_layers.layers import TensorizedLinear_linear, TensorizedLinear_module, TensorizedLinear_module_tonn
from tensor_layers.low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from tensor_fwd_bwd.tensorized_linear import TensorizedLinear
# from tensorized_linear import TensorizedLinear

__all__ = ["ZO_SGD_mask"]

opt_able_layers_dict = {
    'nn.Linear': nn.Linear,
    'nn.Conv2d': nn.Conv2d,
    'TensorizedLinear': TensorizedLinear,
    'TensorizedLinear_module': TensorizedLinear_module,
    'TensorizedLinear_module_tonn': TensorizedLinear_module_tonn
}

class ZO_SGD_mask(Optimizer):
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
        defaults = dict(lr=lr)
        super().__init__(model.parameters(), defaults)
        self.sigma = sigma
        self.n_sample = n_sample
        self.forward_counter = 0
        self.global_step = 0
        self.model = model
        self.criterion = criterion
        self.masks = masks
        self.signSGD = signSGD
        self.layer_by_layer = layer_by_layer
        self.opt_layers_strs = opt_layers_strs

        self.init_state()

    def init_state(self):
        self.opt_layers = self.create_opt_layers()
        self.modules = self.extract_modules(self.model)
        self.trainable_params = self.extract_trainable_parameters(self.model)
        # for layer_name, layer_param in self.trainable_params.items():
        #     print(layer_name)
        #     for param_name, _ in layer_param.items():
        #         print(param_name)
        # self.untrainable_params = self.extract_untrainable_parameters(self.model)
        if self.masks is not None:
            self.enable_mixedtraining(self.masks)
        else:
            self.disable_mixedtraining()

    def create_opt_layers(self):
        if isinstance(self.opt_layers_strs, str):
            return opt_able_layers_dict[self.opt_layers_strs]
        elif isinstance(self.opt_layers_strs, list):
            opt_layers = []
            for layer_str in self.opt_layers_strs:
                opt_layers.append(opt_able_layers_dict[layer_str])
            return tuple(opt_layers)
        else:
            raise (ValueError("opt_layers_strs should either be a string of a list of strings"))
    
    def extract_modules(self, model):
        return {
            layer_name: layer
            for layer_name, layer in model.named_modules()
            if isinstance(layer, self.opt_layers)
        }

    def extract_trainable_parameters(self, model):
    # flatten the parameters
        trainable_parameters = dict()
        for layer_name, layer in self.modules.items():
            if isinstance(layer, (TensorizedLinear_module_tonn)):
                trainable_parameters[layer_name] = {
                    "tt_cores-"+str(i): getattr(layer,'tt_cores')[i].weight.data.view(-1)
                    for i in range(getattr(layer, 'order'))
                }
            elif isinstance(layer, (TensorizedLinear_module)):
                trainable_parameters[layer_name] = {
                    str(i): getattr(layer.tensor,'factors')[i].data.view(-1)
                    for i in range(getattr(layer.tensor, 'order'))
                }
            elif isinstance(layer, (TensorizedLinear)):
                trainable_parameters[layer_name] = {
                    str(i): getattr(layer.weight.factors, 'factor_'+str(i)).data.view(-1)
                    for i in range(getattr(layer.weight, 'order'))
                }
            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                trainable_parameters[layer_name] = {  
                    param_name: getattr(layer, param_name).data.view(-1)
                    for param_name in ["weight"]
                }
        return trainable_parameters

    # def extract_untrainable_parameters(self, model):
    #     return {
    #         layer_name: {
    #             param_name: getattr(layer, param_name)
    #             for param_name in ["phase_bias_U", "phase_bias_V", "delta_list_U", "delta_list_V"]
    #         }
    #         for layer_name, layer in model.named_modules()
    #         if isinstance(layer, (TensorizedLinear_module))
    #     }

    def enable_mixedtraining(self, masks):
        self.mixedtrain_masks = {
            layer_name: {
                p_name: masks[layer_name][p_name].view(-1)
                for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in self.trainable_params.items()
        }
        print(self.mixedtrain_masks)

    def disable_mixedtraining(self):
        self.mixedtrain_masks = {
            layer_name: {
                p_name: torch.ones_like(p, device=p.device) 
                for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in self.trainable_params.items()
        }
    
    def extract_grad_fo(self, model):
    # flatten the parameters
        grad_fo = dict()
        for layer_name, layer in self.modules.items():
            if isinstance(layer, (TensorizedLinear_module_tonn)):
                grad_fo[layer_name] = {
                    "tt_cores-"+str(i): getattr(layer,'tt_cores')[i].weight.grad.view(-1)
                    for i in range(getattr(layer, 'order'))
                }
            elif isinstance(layer, (TensorizedLinear_module)):
                grad_fo[layer_name] = {
                    str(i): getattr(layer.tensor,'factors')[i].grad.view(-1)
                    for i in range(getattr(layer.tensor, 'order'))
                }
            elif isinstance(layer, (TensorizedLinear)):
                grad_fo[layer_name] = {
                    str(i): getattr(layer.weight.factors, 'factor_'+str(i)).grad.view(-1)
                    for i in range(getattr(layer.weight, 'order'))
                }
            elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                grad_fo[layer_name] = {  
                    param_name: getattr(layer, param_name).grad.view(-1)
                    for param_name in ["weight"]
                }
        return grad_fo

    def cal_grad_err(self, params, grad_zo, grad_fo):
        return {
            layer_name: {
                p_name: grad_zo[layer_name][p_name] - grad_fo[layer_name][p_name]
                for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in params.items()
        }

    def _sample_perturbation(self, x, sigma):
        with torch.random.fork_rng():
            torch.random.manual_seed(np.random.randint(0, 1000))
            return torch.randn(x.size(), device=x.device).mul_(sigma)

    # params: trainable_params -> {layer_name: {p_name, p}}
    def perturb(self, params, sigma):
        perturbs = {
            layer_name: {p_name: self._sample_perturbation(p, sigma).mul(self.mixedtrain_masks[layer_name][p_name]) for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }
        # p_name: phase_U phase_S phase_V
        # p: the corresponding tensor (flattened)
        non_perturbed_params = copy.deepcopy(params)
        pos_perturbed_params = {
            layer_name: {p_name: p + perturbs[layer_name][p_name] for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }
        neg_perturbed_params = {
            layer_name: {p_name: p - perturbs[layer_name][p_name] for p_name, p in layer_params.items()}
            for layer_name, layer_params in params.items()
        }
        return perturbs, non_perturbed_params, pos_perturbed_params, neg_perturbed_params

    # params: perturbed_params -> {layer_name: {p_name, p}}
    # ============ commit all of the perturbed params to the model params ============
    def commit_all(self, params) -> None:
        # layer = self.modules[layer_name]
        for layer_name, layer in self.modules.items():
            for param_name, param in params[layer_name].items():   # params[layer_name].items(): {p_name, p}
                if isinstance(layer, (TensorizedLinear_module_tonn)):
                    raise NotImplementedError
                elif isinstance(layer, (TensorizedLinear_module)):
                    idx = int(param_name)
                    ttm_shape = layer.tensor.factors[idx].shape
                    layer.tensor.factors[idx].data = param.reshape(ttm_shape)
                elif isinstance(layer, (TensorizedLinear)):
                    idx = int(param_name)
                    tt_shape = (layer.weight.rank[idx],
                                layer.weight.shape[idx],
                                layer.weight.rank[idx+1])
                    # t_param = torch.nn.parameter.Parameter(param.reshape(tt_shape), requires_grad=False)
                    # setattr(layer.weight.factors, 'factor_'+param_name, t_param)
                    setattr(getattr(layer.weight.factors, 'factor_'+param_name), 'data', param.reshape(tt_shape))
                elif isinstance(layer, (nn.Linear, nn.Conv2d)):
                    if param_name == 'weight':
                        # layer.weight.data = param
                        layer.weight.data = param.reshape(layer.out_features, layer.in_features)
                    else:
                        raise ValueError(f"Wrong param_name {param_name}")

    def _compute_gradient_all(self, loss_diff, perturb, sigma):
        c = 1 / sigma ** 2
        loss_diff = c * loss_diff
        return {
            layer_name: {p_name: p.mul(loss_diff) for p_name, p in layer_params.items()}
            for layer_name, layer_params in perturb.items()
        }

    def _accumulate_gradient_all(self, buffer, grad, scale):
        return {
            layer_name: {
                p_name: p.add_(grad[layer_name][p_name] * scale) for p_name, p in layer_params.items()
            }
            for layer_name, layer_params in buffer.items()
        }

    # ============ commit the param layer by layer ============
    def commit(self, layer_name: str, param_name: str, param: Tensor) -> None:
        '''
            layer_name:{param_name: p}
        '''
        layer = self.modules[layer_name]
        if isinstance(layer, (TensorizedLinear_module_tonn)):
            raise NotImplementedError
        elif isinstance(layer, (TensorizedLinear_module)):
            idx = int(param_name)
            ttm_shape = layer.tensor.factors[idx].shape
            layer.tensor.factors[idx].data = param.reshape(ttm_shape)
        elif isinstance(layer, (TensorizedLinear)):
            idx = int(param_name)
            tt_shape = (layer.weight.rank[idx],
                        layer.weight.shape[idx],
                        layer.weight.rank[idx+1])
            # t_param = torch.nn.parameter.Parameter(param.reshape(tt_shape), requires_grad=False)
            # setattr(layer.weight.factors, 'factor_'+param_name, t_param)
            setattr(getattr(layer.weight.factors, 'factor_'+param_name), 'data', param.reshape(tt_shape))
        elif isinstance(layer, (nn.Linear, nn.Conv2d)):
            if param_name == "weight":
                layer.weight.data = param.reshape(layer.out_features, layer.in_features)
            else:
                raise ValueError(f"Wrong param_name {param_name}")
    
    def _compute_gradient(self, loss_diff, perturb, sigma):
        c = 1 / sigma ** 2
        loss_diff = c * loss_diff

        return perturb.mul(loss_diff)

    def _accumulate_gradient(self, buffer, grad, scale):
        
        return buffer.add_(grad * scale)
    
    # ============ apply gradients all together ============
    def _apply_gradients(self, params, grad, lr):
        if self.signSGD == True:
            return {
                layer_name: {p_name: p.sub_(torch.sign(grad[layer_name][p_name]) * lr) for p_name, p in layer_params.items()}
                for layer_name, layer_params in params.items()
            }
        else:
            return {
                layer_name: {p_name: p.sub_(grad[layer_name][p_name] * lr) for p_name, p in layer_params.items()}
                for layer_name, layer_params in params.items()
            }

    # def zo_gradient_descent_all(self, obj_fn, params):
    #     """
    #     description: stochastic zo gradient descent.
    #     """
    #     # evaluate objective on the current parameters
    #     with torch.no_grad():
    #         y, old_loss = obj_fn()
    #         self.forward_counter += 1
    #     lr = get_learning_rate(self)
    #     grads = None
    #     for _ in range(self.n_sample):
    #         perturb, perturbed_params = self.perturb(params, self.sigma)
    #         with torch.no_grad():  # training=True to enable profiling, but do not save graph
    #             self.commit_all(perturbed_params)
    #             _, new_loss = obj_fn()
    #             self.forward_counter += 1
    #         grad = self._compute_gradient_all(new_loss - old_loss, perturb, self.sigma)
    #         grads = grad if grads is None else self._accumulate_gradient_all(grads, grad, 1 / self.n_sample)
    #     self._apply_gradients(params, grads, lr)

    #     return y, old_loss
    
    def zo_gradient_descent_all(self, obj_fn, params):
        """
        description: stochastic zo gradient descent.
        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)
        grads = None
        for _ in range(self.n_sample):
            perturbs, non_perturbed_params, pos_perturbed_params, neg_perturbed_params = self.perturb(params, self.sigma)
            with torch.no_grad():  # training=True to enable profiling, but do not save graph
                self.commit_all(pos_perturbed_params)
                _, pos_loss = obj_fn()
                self.commit_all(neg_perturbed_params)
                _, neg_loss = obj_fn()
                self.forward_counter += 1
                self.commit_all(non_perturbed_params)
            grad = self._compute_gradient_all((pos_loss-neg_loss)/2, perturbs, self.sigma)
            grads = grad if grads is None else self._accumulate_gradient_all(grads, grad, 1 / self.n_sample)
        self._apply_gradients(params, grads, lr)

        return y, old_loss, grads
    
    def zo_gradient_descent(self, obj_fn, params):
        """
        description: stochastic zo gradient descent.
        """
        # evaluate objective on the current parameters
        with torch.no_grad():
            y, old_loss = obj_fn()
            self.forward_counter += 1
        lr = get_learning_rate(self)
        grads = dict()
        perturbs, non_perturbed_params, pos_perturbed_params, neg_perturbed_params = self.perturb(params, self.sigma)

        # estimate gradients layer-by-layer
        for layer_name, layer in self.modules.items():
            layer_grads = dict()
            
            for p_name, perturb in perturbs[layer_name].items():   # params[layer_name].items(): {p_name, p}
                for _ in range(self.n_sample):
                    param_grad = None
                    with torch.no_grad():  # training=True to enable profiling, but do not save graph
                        self.commit(layer_name, p_name, pos_perturbed_params[layer_name][p_name])
                        _, pos_loss = obj_fn()
                        self.commit(layer_name, p_name, neg_perturbed_params[layer_name][p_name])
                        _, neg_loss = obj_fn()
                        self.forward_counter += 1
                        self.commit(layer_name, p_name, non_perturbed_params[layer_name][p_name])
                    grad = self._compute_gradient((pos_loss-neg_loss)/2, perturb, self.sigma)
                    param_grad = grad if param_grad is None else self._accumulate_gradient(param_grad, grad, 1 / self.n_sample)
                layer_grads[p_name] = param_grad
            
            grads[layer_name] = layer_grads
        self._apply_gradients(params, grads, lr)

        return y, old_loss, grads

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

    def step(self, data, target, en_debug=False, ATIS=False):
        if ATIS == True:
            self.obj_fn = self.build_obj_fn_ATIS(data, target, self.model, self.criterion)
        else:
            self.obj_fn = self.build_obj_fn(data, target, self.model, self.criterion)
        
        if self.layer_by_layer == False:
            y, loss, grads_zo = self.zo_gradient_descent_all(self.obj_fn, self.trainable_params)
        else:
            y, loss, grads_zo = self.zo_gradient_descent(self.obj_fn, self.trainable_params)
        # update internal parameters
        self.global_step += 1
        if en_debug == True:
            grads_fo = self.extract_grad_fo(self.model)
            grads_err = self.cal_grad_err(self.trainable_params, grads_zo, grads_fo)
            return y, loss, (grads_zo, grads_fo, grads_err)
        else:
            return y, loss, grads_zo
