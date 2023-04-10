import torch
import torch.nn as nn
import torch.optim as optim

from optimizer import ZO_SCD_mask, ZO_SGD_mask, ZO_SCD_esti, ZO_SCD_grad
from utils.common_utils import PresetLRScheduler
from models.tensor_models import MNIST_FC, MNIST_CNN, MNIST_TTM, MNIST_TT

def build_model(config):
    if config.GraSP.network == 'fc':
        model = MNIST_FC()
        return model
    elif config.GraSP.network == 'ttm':
        model = MNIST_TTM(
            tensor_type=config.model.tensor_type,
            max_rank = config.model.max_rank
        )
        return model
    elif config.GraSP.network == 'tt':
        model = MNIST_TT(
            tensor_type='tt',
            rank=config.model.rank,
            dropouts=config.model.dropouts
        )
        return model
    else:
        raise ValueError(f"Wrong network_name {config.GraSP.network}")
    
def build_optimizer(config, net, criterion, named_masks, learning_rate, weight_decay):
    if 'ZO_SCD' in config.optimizer.name:
        if config.optimizer.debug == True:
            net.requires_grad_(True)
        else:
            net.requires_grad_(False)

        # if config.optimizer.name == 'ZO_SCD_mask':
        #     grad_estimator = 'sign'
        # elif config.optimizer.name == 'ZO_SCD_grad':
        #     grad_estimator = 'batch'
        # elif config.optimizer.name == 'ZO_SCD_esti':
        #     grad_estimator = 'esti'
        # elif config.optimizer.name == 'ZO_SCD_STP':
        #     grad_estimator = 'STP'
        # else:
        #     raise ValueError(f"Wrong ZO_SCD optimizer name {config.optimizer.name}")

        grad_estimator = config.optimizer.grad_estimator
        
        optimizer = ZO_SCD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rate,
            grad_sparsity = config.optimizer.grad_sparsity,
            h_smooth = config.optimizer.h_smooth if hasattr(config.optimizer, 'h_smooth') else 0.1,
            grad_estimator = grad_estimator,
            opt_layers_strs = config.optimizer.opt_layers_strs,
            STP = config.optimizer.STP if hasattr(config.optimizer, 'STP') else False,
            momentum = config.optimizer.momentum if hasattr(config.optimizer, 'momentum') else 0,
            weight_decay = config.optimizer.weight_decay if hasattr(config.optimizer, 'weight_decay') else 0,
            dampening = config.optimizer.dampening if hasattr(config.optimizer, 'dampening') else 0,
            adam = config.optimizer.adam if hasattr(config.optimizer, 'adam') else False,
            beta_1 = config.optimizer.beta_1 if hasattr(config.optimizer, 'beta_1') else 0.9,
            beta_2 = config.optimizer.beta_2 if hasattr(config.optimizer, 'beta_2') else 0.98
        )
        return optimizer
    elif config.optimizer.name == 'ZO_SGD_mask':
        if config.optimizer.debug == True:
            net.requires_grad_(True)
        else:
            net.requires_grad_(False)
        optimizer = ZO_SGD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rate,
            sigma = config.optimizer.sigma,
            n_sample  = config.optimizer.n_sample,
            signSGD = config.optimizer.signSGD,
            layer_by_layer = config.optimizer.layer_by_layer,
            opt_layers_strs = config.optimizer.opt_layers_strs
        )
        return optimizer
    # elif config.optimizer.name == 'ZO_SCD_esti':
    #     net.requires_grad_(False)
    #     optimizer = ZO_SCD_esti(
    #         model = net, 
    #         criterion = criterion,
    #         masks = named_masks,
    #         lr = learning_rate,
    #         grad_sparsity = config.optimizer.grad_sparsity,
    #         tensorized = config.model.tensorized,
    #         h_smooth = config.optimizer.h_smooth
    #     )
    #     return optimizer
    # elif config.optimizer.name == 'ZO_SCD_grad':
    #     net.requires_grad_(False)
    #     optimizer = ZO_SCD_grad(
    #         model = net, 
    #         criterion = criterion,
    #         masks = named_masks,
    #         lr = learning_rate,
    #         grad_sparsity = config.optimizer.grad_sparsity,
    #         tensorized = config.model.tensorized
    #     )
    #     return optimizer
    elif config.optimizer.name == 'ZO_mix':
        optimizer_SGD = ZO_SGD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rate,
            sigma = config.optimizer.sigma,
            n_sample  = config.optimizer.n_sample,
            signSGD = config.optimizer.signSGD,
            layer_by_layer = config.optimizer.layer_by_layer,
            tensorized = config.model.tensorized
        )
        optimizer_SCD = ZO_SCD_mask(
            model = net, 
            criterion = criterion,
            masks = named_masks,
            lr = config.optimizer.SCD_lr,
            grad_sparsity = config.optimizer.grad_sparsity,
            tensorized = config.model.tensorized
        )
        return optimizer_SGD, optimizer_SCD
    elif config.optimizer.name == 'SGD':
        net.requires_grad_(True)
        optimizer = optim.SGD(net.parameters(), lr=learning_rate)
        # optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
        return optimizer
    elif config.optimizer.name == 'ADAM':
        net.requires_grad_(True)
        # optimizer = optim.Adam(list(net.parameters()), lr=learning_rate, weight_decay=weight_decay)
        optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-06, lr = learning_rate, weight_decay=weight_decay)
        return optimizer
    else:
        raise ValueError(f"Wrong optimizer_name {config.optimizer.name}") 

def build_scheduler(config, optimizers, learning_rate):
    if config.scheduler.name == 'PresetLRScheduler':
        lr_schedule = dict()
        for n_epoch,lr_coef in dict(config.scheduler.lr_schedule).items():
            lr_schedule[n_epoch] = learning_rate * lr_coef
    
        lr_scheduler = PresetLRScheduler(lr_schedule)
        return lr_scheduler
    elif config.scheduler.name == 'ExponentialLR':
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizers, gamma=config.scheduler.gamma)
        return lr_scheduler
    elif config.scheduler.name == 'ZO_mix':
        lr_scheduler_SGD = optim.lr_scheduler.ExponentialLR(optimizers[0], gamma=config.scheduler.gamma)
        lr_schedule = dict()
        for n_epoch,lr_coef in dict(config.scheduler.lr_schedule).items():
            lr_schedule[n_epoch] = config.optimizer.SCD_lr * lr_coef
    
        lr_scheduler_SCD = PresetLRScheduler(lr_schedule)

        return lr_scheduler_SGD, lr_scheduler_SCD

    else:
        raise ValueError(f"Wrong scheduler_name {config.scheduler.name}")