import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
import types

from tensor_layers.layers import TensorizedLinear_module


def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
    datas = [[] for _ in range(num_classes)]
    labels = [[] for _ in range(num_classes)]
    mark = dict()
    dataloader_iter = iter(dataloader)
    while True:
        inputs, targets = next(dataloader_iter)
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(datas[category]) == samples_per_class:
                mark[category] = True
                continue
            datas[category].append(x)
            labels[category].append(y)
        if len(mark) == num_classes:
            break

    X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
    return X, y


def count_total_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            total += m.weight.numel()
    return total


def count_fc_parameters(net):
    total = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear)):
            total += m.weight.numel()
    return total


def GraSP_newHg(net, ratio, train_dataloader, device, 
                  num_classes=10, samples_per_class=25, num_iters=1, T=200, 
                  reinit=True, tensorized=False):
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []
    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)

    # ================== Extract model parameters ==================
    if tensorized == 'TensorizedLinear_module':
        print('GraSP on TensorizedLinear_module')
        for layer in net.modules():
            if isinstance(layer, TensorizedLinear_module):
                for i in range(layer.tensor.order):
                    weights.append(layer.tensor.factors[i])
    else:
        # rescale_weights(net)
        print('GraSP on nn.Linear and nn.Conv2d')
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                if isinstance(layer, nn.Linear) and reinit:
                    nn.init.xavier_normal(layer.weight)
                weights.append(layer.weight)
    

    inputs_one = []
    targets_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        # stop_grad(p)
        inputs, targets = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        inputs = inputs.to(device)
        targets = targets.to(device)

        outputs = net.forward(inputs)/T
        loss = F.cross_entropy(outputs, targets)
        
        grads = torch.autograd.grad(loss, weights, create_graph=True)
        # Compute the Hessian-vector product (HVP) for each parameter and gradient pair
        hessian_grad_products = []
        for grad in grads:
            grad_flat = grad.view(-1)
            hvp = torch.autograd.grad(grad_flat, weights, grad_outputs=inputs, retain_graph=True)
            hessian_grad_products.extend(hvp)
    
    grads = dict()
    named_grads = dict()
    old_modules = list(old_net.modules())
    for idx, (layer_name, layer) in enumerate(net.named_modules()):
        if tensorized == 'TensorizedLinear_module':
            if isinstance(layer, TensorizedLinear_module):
                layer_grads = {
                    str(i): -layer.tensor.factors[i].data * layer.tensor.factors[i].grad
                    for i in range(layer.tensor.order)
                }
                grads[old_modules[idx]] = layer_grads
                named_grads[layer_name] = layer_grads
        else: 
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg
                named_grads[layer_name] = -layer.weight.data * layer.weight.grad

    # =============== for ZO optimizer ===============
    # named_grads = dict()
    # for layer_name, layer in net.named_modules():
    #     if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
    #         named_grads[layer_name] = -layer.weight.data * layer.weight.grad  # -theta_q Hg
    
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat(
        [ torch.flatten(x) if torch.is_tensor(x)
          else torch.cat(
            [torch.flatten(x_v) for x_v in x.values()]
          )
          for x in grads.values()]
    )
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for layer, g in grads.items():
        if tensorized == 'TensorizedLinear_module':
            layer_mask = dict()
            for g_name, g_factor in g.items():
                layer_mask[g_name] = ((g_factor / norm_factor) <= acceptable_score).float()
            keep_masks[layer] = layer_mask
        else:
            keep_masks[layer] = ((g / norm_factor) <= acceptable_score).float()
    
    # =============== for ZO optimizer ===============
    named_keep_masks = dict()
    for layer_name, g in named_grads.items():
        if tensorized == 'TensorizedLinear_module':
            layer_mask = dict()
            for g_name, g_factor in g.items():
                layer_mask[g_name] = ((g_factor / norm_factor) <= acceptable_score).float()
            named_keep_masks[layer_name] = layer_mask
        else:
            named_keep_masks[layer_name] = {'weight': ((g / norm_factor) <= acceptable_score).float()}

    # print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks, named_keep_masks