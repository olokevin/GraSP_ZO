import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math

import copy
import types


# def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
#     datas = [[] for _ in range(num_classes)]
#     labels = [[] for _ in range(num_classes)]
#     mark = dict()
#     dataloader_iter = iter(dataloader)
#     while True:
#         inputs, targets = next(dataloader_iter)
#         # inputs.shape[0]: batch_sz
#         for idx in range(inputs.shape[0]):
#             # x,y are tensors with size 1 (select the inputs[idx] and turn to a tensor list)
#             x, y = inputs[idx:idx+1], targets[idx:idx+1]
#             category = y.item()
#             if len(datas[category]) == samples_per_class:
#                 mark[category] = True
#                 continue
#             datas[category].append(x)
#             labels[category].append(y)
#         if len(mark) == num_classes:
#             break

#     X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
#     return X, y

# def GraSP_fetch_data(dataloader, num_classes, samples_per_class):
#     datas = [[] for _ in range(num_classes)]
#     labels = [[] for _ in range(num_classes)]
#     slot_labels = [[] for _ in range(num_classes)]
#     attns = [[] for _ in range(num_classes)]
#     segs = [[] for _ in range(num_classes)]

#     cnts = torch.zeros(num_classes)
#     mark = dict()
#     dataloader_iter = iter(dataloader)
#     while True:
#         target, w1, slot_label,attn,seg = next(dataloader_iter)
#         for idx in range(w1.shape[0]):
#             x, y, t_slot_label,t_attn,t_seg = w1[idx:idx+1], target[idx:idx+1], slot_label[idx:idx+1], attn[idx:idx+1], seg[idx:idx+1]
#             # category = y.item()
#             # in ATIS label range from 1~21
#             category = y.item()-1
#             cnts[category] = cnts[category] + 1
#             if len(datas[category]) == samples_per_class:
#                 mark[category] = True
#                 continue
#             datas[category].append(x)
#             labels[category].append(y)
#             slot_labels[category].append(t_slot_label)
#             attns[category].append(t_attn)
#             segs[category].append(t_seg)
#         if len(mark) == num_classes:
#             break

#     X, y = torch.cat([torch.cat(_, 0) for _ in datas]), torch.cat([torch.cat(_) for _ in labels]).view(-1)
#     slot_labels = torch.cat([torch.cat(_) for _ in labels]).view(-1)
#     attns= torch.cat([torch.cat(_, 0) for _ in attns])
#     segs = torch.cat([torch.cat(_, 0) for _ in segs])
#     return X, y, slot_labels, attns, segs

# def GraSP_fetch_data(dataloader, samples_batches):
#     datas = []
#     labels = []
#     slot_labels = []
#     attns = []
#     segs = []
#     mark = dict()
#     dataloader_iter = iter(dataloader)
#     for idx in range(samples_batches):
#         target, w1, slot_label,attn,seg = next(dataloader_iter)
#         if idx == 0:
#             datas = w1
#             labels = target
#             slot_labels = slot_label
#             attns = attn
#             segs = seg
#         else:
#             datas = torch.cat((datas,w1), 0)
#             labels = torch.cat((labels,target), 0)
#             slot_labels = torch.cat((slot_labels,slot_label), 0)
#             attns = torch.cat((attns,attn), 0)
#             segs = torch.cat((segs,seg), 0)
    
#     return datas, labels, slot_labels, attns, segs


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


# def GraSP_attn(net, ratio, train_dataloader, device, num_classes=10, samples_per_class=25, num_iters=1, T=200, reinit=True):
def GraSP_attn(net, ratio, sample_dataloader, device, num_iters=1, T=200, reinit=True):
    # T: temperature, to smooth softmax
    eps = 1e-10
    keep_ratio = 1-ratio
    old_net = net

    net = copy.deepcopy(net)  # .eval()
    net.zero_grad()

    weights = []
    total_parameters = count_total_parameters(net)
    fc_parameters = count_fc_parameters(net)

    # rescale_weights(net)
    for layer in net.modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            if isinstance(layer, nn.Linear) and reinit:
                nn.init.xavier_normal(layer.weight)
            weights.append(layer.weight)

    inputs_one = []
    targets_one = []
    slot_label_one = []
    attn_one = []
    seg_one = []

    grad_w = None
    for w in weights:
        w.requires_grad_(True)

    print_once = False
    for it in range(num_iters):
        print("(1): Iterations %d/%d." % (it, num_iters))
        # =========== sample dataset ===========
        # inputs, targets, slot_label, attn, seg = GraSP_fetch_data(train_dataloader, samples_batches)
        # inputs, targets, slot_label, attn, seg = GraSP_fetch_data(train_dataloader, num_classes, samples_per_class)
        
        dataloader_iter = iter(sample_dataloader)
        targets, inputs, slot_label,attn,seg = next(dataloader_iter)
        
        N = inputs.shape[0]

        din = copy.deepcopy(inputs)
        inputs_one.append(din[:N//2])
        inputs_one.append(din[N // 2:])
        
        dtarget = copy.deepcopy(targets)
        targets_one.append(dtarget[:N//2])
        targets_one.append(dtarget[N // 2:])

        d_slot_label = copy.deepcopy(slot_label)
        slot_label_one.append(d_slot_label[:N//2])
        slot_label_one.append(d_slot_label[N // 2:])

        d_attn = copy.deepcopy(attn)
        attn_one.append(d_attn[:N//2])
        attn_one.append(d_attn[N // 2:])

        d_seg = copy.deepcopy(seg)
        seg_one.append(d_seg[:N//2])
        seg_one.append(d_seg[N // 2:])

        # inputs_one, targets_one: list, containing two
        inputs = inputs.to(device)
        targets = targets.to(device)
        slot_label = slot_label.to(device)
        attn = attn.to(device)
        seg = seg.to(device)
        
        # =========== one step forward, FIRST HALF ===========
        Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
        pred,pred_slot = net(inputs[:N//2],attn=attn[:N//2],seg=seg[:N//2])
        pred = pred / T
        pred_slot = pred_slot / T
        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
        
        t_slot_label = torch.flatten(slot_label[:N//2],start_dim=0, end_dim=1)
        loss_MLM =  Loss(pred_slot, t_slot_label)

        loss = Loss(pred,targets[:N//2]) + loss_MLM

        # slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)
        # loss_MLM =  Loss(pred_slot, slot_label)
        # loss = Loss(pred,targets)  + loss_MLM

        # outputs = net.forward(inputs[:N//2])/T
        # if print_once:
        #     # import pdb; pdb.set_trace()
        #     x = F.softmax(outputs)
        #     print(x)
        #     print(x.max(), x.min())
        #     print_once = False
        # loss = F.cross_entropy(outputs, targets[:N//2])
        # ===== debug ================
        grad_w_p = autograd.grad(loss, weights)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

        # =========== one step forward, SECOND HALF ===========
        pred,pred_slot = net(inputs[N // 2:],attn=attn[N // 2:],seg=seg[N // 2:])
        pred = pred / T
        pred_slot = pred_slot / T
        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)

        t_slot_label = torch.flatten(slot_label[N // 2:],start_dim=0, end_dim=1)
        loss_MLM =  Loss(pred_slot, t_slot_label)
        # loss = F.cross_entropy(outputs, targets[N // 2:])
        loss = Loss(pred,targets[N // 2:]) + loss_MLM
        grad_w_p = autograd.grad(loss, weights, create_graph=False)
        if grad_w is None:
            grad_w = list(grad_w_p)
        else:
            for idx in range(len(grad_w)):
                grad_w[idx] += grad_w_p[idx]

    # ret_inputs = []
    # ret_targets = []

    for it in range(len(inputs_one)):
        print("(2): Iterations %d/%d." % (it, num_iters))
        inputs = inputs_one.pop(0).to(device)
        targets = targets_one.pop(0).to(device)
        slot_label = slot_label_one.pop(0).to(device)
        attn = attn_one.pop(0).to(device)
        seg = seg_one.pop(0).to(device)
        # ret_inputs.append(inputs)
        # ret_targets.append(targets)
        
        # outputs = net.forward(inputs)/T
        pred,pred_slot = net(inputs,attn=attn,seg=seg)
        pred = pred / T
        pred_slot = pred_slot / T
        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)

        t_slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)
        loss_MLM =  Loss(pred_slot, t_slot_label)

        # loss = F.cross_entropy(outputs, targets)
        loss = Loss(pred,targets) + loss_MLM
        # ===== debug ==============

        grad_f = autograd.grad(loss, weights, create_graph=True)
        z = 0
        count = 0
        for layer in net.modules():
            if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                z += (grad_w[count].data * grad_f[count]).sum()
                count += 1
        z.backward()

    grads = dict()
    old_modules = list(old_net.modules())
    for idx, layer in enumerate(net.modules()):
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            grads[old_modules[idx]] = -layer.weight.data * layer.weight.grad  # -theta_q Hg

    # =============== for ZO optimizer ===============
    named_grads = dict()
    for layer_name, layer in net.named_modules():
        if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
            named_grads[layer_name] = -layer.weight.data * layer.weight.grad  # -theta_q Hg
    
    # Gather all scores in a single vector and normalise
    all_scores = torch.cat([torch.flatten(x) for x in grads.values()])
    norm_factor = torch.abs(torch.sum(all_scores)) + eps
    print("** norm factor:", norm_factor)
    all_scores.div_(norm_factor)

    num_params_to_rm = int(len(all_scores) * (1-keep_ratio))
    threshold, _ = torch.topk(all_scores, num_params_to_rm, sorted=True)
    # import pdb; pdb.set_trace()
    acceptable_score = threshold[-1]
    print('** accept: ', acceptable_score)
    keep_masks = dict()
    for m, g in grads.items():
        keep_masks[m] = ((g / norm_factor) <= acceptable_score).float()

    # =============== for ZO optimizer ===============
    named_keep_masks = dict()
    for m, g in named_grads.items():
        named_keep_masks[m] = {'weight': ((g / norm_factor) <= acceptable_score).float()}
    
    print(torch.sum(torch.cat([torch.flatten(x == 1) for x in keep_masks.values()])))

    return keep_masks, named_keep_masks
