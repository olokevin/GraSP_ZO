import argparse
import json
import math
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from utils.builder import build_optimizer, build_scheduler
from models.model_base import ModelBase
from pruner.GraSP_zo_mask import GraSP_zo_mask

from pyutils.config import configs
from optimizer import ZO_SCD_mask, ZO_SGD_mask

# from tensor_layers.layers import TensorizedLinear_module
# from tensor_fwd_bwd.tensorized_linear import TensorizedLinear

from models.tensor_models import MNIST_FC, MNIST_CNN, MNIST_TTM, MNIST_TT

# disabled
def init_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--run', type=str, default='')
    args = parser.parse_args()
    runs = None
    if len(args.run) > 0:
        runs = args.run
    config = process_config(args.config, runs)

    return config


def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/base/%s.py' % config.network.lower())
    path_main = os.path.join(path, 'main_prune_non_imagenet.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
    logger = get_logger('log', logpath=config.summary_dir + '/',
                        filepath=path_model, package_files=[path_main, path_pruner])
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    # sys.stdout = open(os.path.join(config.summary_dir, 'stdout.txt'), 'w+')
    # sys.stderr = open(os.path.join(config.summary_dir, 'stderr.txt'), 'w+')
    return logger, writer

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        count += 1


def save_state(net, acc, epoch, loss, config, ckpt_path, is_best=False):
    print('Saving..')
    state = {
        'net': net,
        'acc': acc,
        'epoch': epoch,
        'loss': loss,
        'args': config
    }
    if not is_best:
        torch.save(state, '%s/pruned_%s_%s%s_%d.t7' % (ckpt_path,
                                                       config.dataset,
                                                       config.network,
                                                       config.depth,
                                                       epoch))
    else:
        torch.save(state, '%s/finetuned_%s_%s%s_best.t7' % (ckpt_path,
                                                            config.dataset,
                                                            config.network,
                                                            config.depth))


def train(net, loader, optimizer, criterion, lr_scheduler, epoch, writer, iteration, logger):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0

    if isinstance(lr_scheduler, PresetLRScheduler):
        lr_scheduler(optimizer, epoch)
        now_lr = lr_scheduler.get_lr(optimizer)
    else:
        now_lr = optimizer.state_dict()['param_groups'][0]['lr']

    
    desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
            (now_lr, 0, 0, correct, total))

    writer.add_scalar('iter_%d/train/lr' % iteration, now_lr, epoch)

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    for batch_idx, (inputs, targets) in prog_bar:
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()

        if isinstance(optimizer, ZO_SCD_mask):
            outputs, loss = optimizer.step(inputs, targets)
        elif isinstance(optimizer, ZO_SGD_mask):
            outputs, loss, grads_e = optimizer.step(inputs, targets)
            # test, check real grads
            if configs.optimizer.debug == True:
                test_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
                outputs = net(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                test_optimizer.step()
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        desc = ('[LR=%s] Loss: %.3f | Acc: %.3f%% (%d/%d)' %
                (now_lr, train_loss / (batch_idx + 1), 100. * correct / total, correct, total))
        prog_bar.set_description(desc, refresh=True)

        # logger.info('epoch[%d], batch_idx[%d], train_loss acc: %.4f' % (epoch, batch_idx, train_loss))

    train_loss = train_loss / (batch_idx + 1)
    train_acc  = 100. * correct / total
        
    logger.info('epoch[%d], lr: %.4f, train_loss: %.4f, train_acc: %.4f' % (epoch, now_lr, train_loss, train_acc))
        
    writer.add_scalar('iter_%d/train/loss' % iteration, train_loss, epoch)
    writer.add_scalar('iter_%d/train/acc' % iteration, train_acc, epoch)


def test(net, loader, criterion, epoch, writer, iteration, logger):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
            % (test_loss / (0 + 1), 0, correct, total))

    prog_bar = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True)
    with torch.no_grad():
        for batch_idx, (inputs, targets) in prog_bar:
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            desc = ('Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss / (batch_idx + 1), 100. * correct / total, correct, total))
            prog_bar.set_description(desc, refresh=True)

    # Save checkpoint.
    test_loss = test_loss / (batch_idx + 1)
    test_acc = 100. * correct / total

    logger.info('epoch[%d], test_loss: %.4f, test_acc: %.4f' % (epoch, test_loss, test_acc))

    writer.add_scalar('iter_%d/test/loss' % iteration, test_loss, epoch)
    writer.add_scalar('iter_%d/test/acc' % iteration, test_acc, epoch)
    return test_acc


def train_once(mb, net, named_masks, trainloader, testloader, writer, config, ckpt_path, learning_rate, weight_decay, num_epochs,
               iteration, logger):
    criterion = nn.CrossEntropyLoss(reduction='mean')
    
    # ================== optimizer ======================
    optimizers = build_optimizer(config, net, criterion, named_masks, learning_rate, weight_decay)
    # if config.optimizer.name == 'ZO_SCD_mask':
    #     net.requires_grad_(False)
    #     optimizer = ZO_SCD_mask(
    #         model = net, 
    #         criterion = criterion,
    #         masks = named_masks,
    #         lr = learning_rate,
    #         grad_sparsity = config.optimizer.grad_sparsity,
    #         tensorized = config.model.tensorized
    #     )
    # elif config.optimizer.name == 'ZO_SGD_mask':
    #     if config.optimizer.debug == True:
    #         net.requires_grad_(True)
    #     else:
    #         net.requires_grad_(False)
    #     optimizer = ZO_SGD_mask(
    #         model = net, 
    #         criterion = criterion,
    #         masks = named_masks,
    #         lr = learning_rate,
    #         sigma = config.optimizer.sigma,
    #         n_sample  = config.optimizer.n_sample,
    #         signSGD = config.optimizer.signSGD,
    #         layer_by_layer = config.optimizer.layer_by_layer,
    #         tensorized = config.model.tensorized
    #     )
    # elif config.optimizer.name == 'ZO_mix':
    #     optimizer_SGD = ZO_SGD_mask(
    #         model = net, 
    #         criterion = criterion,
    #         masks = named_masks,
    #         lr = learning_rate,
    #         sigma = config.optimizer.sigma,
    #         n_sample  = config.optimizer.n_sample,
    #         signSGD = config.optimizer.signSGD,
    #         layer_by_layer = config.optimizer.layer_by_layer,
    #         tensorized = config.model.tensorized
    #     )
    #     optimizer_SCD = ZO_SCD_mask(
    #         model = net, 
    #         criterion = criterion,
    #         masks = named_masks,
    #         lr = config.optimizer.SCD_lr,
    #         grad_sparsity = config.optimizer.grad_sparsity,
    #         tensorized = config.model.tensorized
    #     )
    #     optimizer = optimizer_SGD
    # elif config.optimizer.name == 'SGD':
    #     optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=weight_decay)
    # elif config.optimizer.name == 'ADAM':
    #     # optimizer = optim.Adam(list(net.parameters()), lr=learning_rate, weight_decay=weight_decay)
    #     optimizer = optim.Adam(net.parameters(), betas=(0.9, 0.98), eps=1e-06, lr = learning_rate)
    # else:
    #     raise ValueError(f"Wrong optimizer_name {config.optimizer.name}")    
    
    # ================== scheduler ======================
    
    lr_schedulers = build_scheduler(config, optimizers, learning_rate)

    # if config.scheduler.name == 'PresetLRScheduler':
    #     lr_schedule = dict()
    #     for n_epoch,lr_coef in dict(config.scheduler.lr_schedule).items():
    #         lr_schedule[n_epoch] = learning_rate * lr_coef
    
    #     lr_scheduler = PresetLRScheduler(lr_schedule)
    # elif config.scheduler.name == 'ExponentialLR':
    #     lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler.gamma)
    # elif config.scheduler.name == 'ZO_mix':
    #     lr_scheduler_SGD = optim.lr_scheduler.ExponentialLR(optimizer, gamma=config.scheduler.gamma)
    #     lr_schedule = dict()
    #     for n_epoch,lr_coef in dict(config.scheduler.lr_schedule).items():
    #         lr_schedule[n_epoch] = config.optimizer.SCD_lr * lr_coef
    
    #     lr_scheduler_SCD = PresetLRScheduler(lr_schedule)

    #     lr_scheduler = lr_scheduler_SGD
    # else:
    #     raise ValueError(f"Wrong scheduler_name {config.scheduler.name}")    
    
    best_acc = 0
    best_epoch = 0
    
    # ================== training epochs ======================

    for epoch in range(num_epochs):
        # mix training selection
        if config.optimizer.name == 'ZO_mix':
            if epoch < config.optimizer.switch_epoch:
                optimizer = optimizers[0]
                lr_scheduler = lr_schedulers[0]
            else:
                optimizer = lr_schedulers[1]
                lr_scheduler = lr_schedulers[1]
        # single training
        else:
            optimizer = optimizers
            lr_scheduler = lr_schedulers

        train(net, trainloader, optimizer, criterion, lr_scheduler, epoch, writer, iteration=iteration, logger=logger)
        test_acc = test(net, testloader, criterion, epoch, writer, iteration, logger=logger)
        
        if isinstance(lr_scheduler, PresetLRScheduler):
            pass
        else:
            lr_scheduler.step()

        if test_acc > best_acc:
            print('Saving..')
            state = {
                'net': net,
                'acc': test_acc,
                'epoch': epoch,
                'args': config.GraSP,
                'mask': mb.masks,
                'ratio': mb.get_ratio_at_each_layer()
            }
            path = os.path.join(ckpt_path, 'finetune_%s_%s%s_r%s_it%d_best.pth.tar' % (config.GraSP.dataset,
                                                                                       config.GraSP.network,
                                                                                       config.GraSP.depth,
                                                                                       config.GraSP.target_ratio,
                                                                                       iteration))
            torch.save(state, path)
            best_acc = test_acc
            best_epoch = epoch
    logger.info('Iteration [%d], best acc: %.4f, epoch: %d' %
                (iteration, best_acc, best_epoch))


def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)

def main():
    # ================== .yml parser ==========================
    # Config: path of *.yml configuration file
    # Currently disabled, and added at front
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-config', metavar='FILE', help='config file')
    args = parser.parse_args()

    # load config *.yml file
    # recursive: also load default.yaml
    configs.load(args.config, recursive=False)  

    print(type(configs.GraSP.pruner))
    
    if configs.GraSP.network == 'fc':
        model = MNIST_FC()
    elif configs.GraSP.network == 'ttm':
        model = MNIST_TTM(
            tensor_type=configs.model.tensor_type,
            max_rank = configs.model.max_rank
        )
    elif configs.GraSP.network == 'tt':
        model = MNIST_TT(
            tensor_type='tt',
            rank=configs.model.rank,
            dropouts=configs.model.dropouts
        )
    else:
        raise ValueError(f"Wrong network_name {configs.GraSP.network}")

    # ================== Prepare logger ==========================
    paths = [configs.GraSP.dataset]
    summn = [configs.GraSP.network, configs.optimizer.name, configs.GraSP.exp_name]
    chekn = [configs.GraSP.network, configs.optimizer.name, configs.GraSP.exp_name]
    if configs.run.runs is not None:
        summn.append('run_%s' % configs.run.runs)
        chekn.append('run_%s' % configs.run.runs)
    summn.append("summary/")
    chekn.append("checkpoint/")
    summary_dir = ["./runs/pruning"] + paths + summn
    ckpt_dir = ["./runs/pruning"] + paths + chekn
    configs.GraSP.summary_dir = os.path.join(*summary_dir)
    configs.GraSP.checkpoint_dir = os.path.join(*ckpt_dir)
    print("=> config.summary_dir:    %s" % configs.GraSP.summary_dir)
    print("=> config.checkpoint_dir: %s" % configs.GraSP.checkpoint_dir)

    logger, writer = init_logger(configs.GraSP)
    logger.info(dict(configs))
    
    # ====================================== graph and stat ======================================
    # t_batch = next(iter(training_data))
    # targets, inputs, slot_label,attn,seg = map(lambda x: x.to(device), t_batch)
    # writer.add_graph(model, (inputs,attn,seg))
    # writer.close()
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())

    print(get_parameter_number(model))

    # ====================================== build ModelBase ======================================
    mb = ModelBase(configs.GraSP.network, configs.GraSP.depth, configs.GraSP.dataset, model)
    mb.cuda()

    # preprocessing
    # ====================================== get dataloader ======================================
    trainloader, testloader = get_dataloader(configs.GraSP.dataset, configs.GraSP.batch_size, 256, 4)
    
    # ====================================== fetch configs ======================================
    ckpt_path = configs.GraSP.checkpoint_dir
    num_iterations = configs.GraSP.iterations
    target_ratio = configs.GraSP.target_ratio
    normalize = configs.GraSP.normalize

    # ====================================== fetch exception ======================================
    # exception = get_exception_layers(mb.model, str_to_list(configs.GraSP.exception, ',', int))
    # logger.info('Exception: ')

    # for idx, m in enumerate(exception):
    #     logger.info('  (%d) %s' % (idx, m))

    # ====================================== fetch training schemes ======================================
    ratio = 1 - (1 - target_ratio) ** (1.0 / num_iterations)
    # learning_rates = str_to_list(config.learning_rate, ',', float)
    # weight_decays = str_to_list(config.weight_decay, ',', float)
    # training_epochs = str_to_list(config.epoch, ',', int)
    learning_rates = float(configs.GraSP.learning_rate)
    weight_decays = float(configs.GraSP.weight_decay)
    training_epochs = int(configs.GraSP.epoch)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))

    # ====================================== start pruning ======================================
    iteration = 0
    if configs.GraSP.pruner != False:
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    1,
                                                                                    num_iterations))

        mb.model.apply(weights_init)
        print("=> Applying weight initialization(%s)." % configs.GraSP.get('init_method', 'kaiming'))
        print("Iteration of: %d/%d" % (iteration, num_iterations))

        masks, named_masks = GraSP_zo_mask(mb.model, ratio, trainloader, 'cuda',
                      num_classes=configs.GraSP.num_classes,
                      samples_per_class=configs.GraSP.samples_per_class,
                      num_iters=configs.GraSP.num_iters)
        iteration = 0
        print('=> Using GraSP')
        # ========== register mask ==================
        mb.register_mask(masks)
        # ========== save pruned network ============
        logger.info('Saving..')
        state = {
            'net': mb.model,
            'acc': -1,
            'epoch': -1,
            'args': configs.GraSP,
            'mask': mb.masks,
            'ratio': mb.get_ratio_at_each_layer()
        }
        path = os.path.join(ckpt_path, 'prune_%s_%s%s_r%s_it%d.pth.tar' % (configs.GraSP.dataset,
                                                                            configs.GraSP.network,
                                                                            configs.GraSP.depth,
                                                                            configs.GraSP.target_ratio,
                                                                            iteration))
        torch.save(state, path)

        # ========== print pruning details ============
        logger.info('**[%d] Mask and training setting: ' % iteration)
        print_mask_information(mb, logger)
        # logger.info('  LR: %.5f, WD: %.5f, Epochs: %d' %
        #             (learning_rates[iteration], weight_decays[iteration], training_epochs[iteration]))
    else:
        named_masks = None
    # ======================= Training =======================
    train_once( mb=mb,
                net=mb.model,
                named_masks = named_masks,
                trainloader=trainloader,
                testloader=testloader,
                writer=writer,
                config=configs,
                ckpt_path=ckpt_path,
              #  learning_rate=learning_rates[iteration],
              #  weight_decay=weight_decays[iteration],
              #  num_epochs=training_epochs[iteration],
                learning_rate=learning_rates,
                weight_decay=weight_decays,
                num_epochs=training_epochs,
                iteration=iteration,
                logger=logger)


if __name__ == '__main__':
    main()
    # config = init_config()
    # main(config)
