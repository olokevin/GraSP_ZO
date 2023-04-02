import argparse
import json
import math
import os
import sys
import time
import shutil
import torch
import torch.nn as nn
import torch.optim as optim

from tensorboardX import SummaryWriter
from tqdm import tqdm
from utils.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.data_utils import get_dataloader
from utils.network_utils import get_network
from utils.builder import build_model, build_optimizer, build_scheduler
from models.model_base import ModelBase
from pruner.GraSP_zo_mask import GraSP_zo_mask

from pyutils.config import configs
from pyutils.torch_train import (
    get_learning_rate,
    get_random_state,
    set_torch_deterministic,
    set_torch_stochastic,
)
from optimizer import ZO_SCD_mask, ZO_SGD_mask, ZO_SCD_esti, ZO_SCD_grad

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
    # makedirs(config.summary_dir)
    # makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'models/base/%s.py' % config.network.lower())
    path_main = os.path.join(path, 'main_prune_non_imagenet.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
    # logger = get_logger('log', logpath=config.summary_dir + '/',
    #                     filepath=path_model, package_files=[path_main, path_pruner])
    logger = get_logger('log', logpath=config.summary_dir,
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
        if isinstance(v, dict):
            for idx, v_ratio in enumerate(v.values()):
                logger.info('  (%d) tt-core %d: Remaining: %.2f%%' % (count, idx, v_ratio))
        else:
            v_ratio = v
            logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v_ratio))
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
        # epoch wise decay
        if isinstance(lr_scheduler, optim.lr_scheduler.ExponentialLR):
            if hasattr(configs.scheduler, 'epoch_wise') and configs.scheduler.epoch_wise == True:
                lr_scheduler.step()
        
        inputs, targets = inputs.cuda(), targets.cuda()
        optimizer.zero_grad()
        # outputs = net(inputs)
        # loss = criterion(outputs, targets)
        # loss.backward()

        # test, check real grads
        en_debug = configs.optimizer.debug if hasattr(configs.optimizer,'debug') else False
        if en_debug == True:
            # test_optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            # test_optimizer.step()

        # compute gradient and update parameters
        if isinstance(optimizer, (ZO_SCD_mask)):
            with torch.no_grad():
                outputs, loss, grads = optimizer.step(inputs, targets, en_debug=en_debug)
        elif isinstance(optimizer, ZO_SGD_mask):
            with torch.no_grad():
                outputs, loss, grads = optimizer.step(inputs, targets, en_debug=en_debug)
        else:
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
        
        if en_debug == True:
            grads_err = grads[2]
            grads_err_norm = dict()
            for layer_name, layer_params in grads_err.items():
                grads_err_norm[layer_name] = dict()
                for p_name, p in layer_params.items():
                    p_norm = torch.linalg.norm(p)
                    grads_err_norm[layer_name][p_name] = p_norm
                    logger.info('epoch[%d], layer: %s, param: %s, grads_err_norm: %.4f' % (epoch, layer_name, p_name, p_norm))
            
            grads_path = os.path.join(*['./figs', configs.GraSP.network, configs.optimizer.name, configs.GraSP.exp_name+'/'])
            makedirs(grads_path)

            if isinstance(optimizer, ZO_SCD_mask):
                grads_path = os.path.join(grads_path, 'h_'+str(configs.optimizer.h_smooth)+'.pth')
                # grads_path = os.path.join(configs.GraSP.summary_dir + 'h_'+str(configs.optimizer.h_smooth)+'.pth')
            elif isinstance(optimizer, ZO_SGD_mask):
                grads_path = os.path.join(grads_path, 'N_'+str(configs.optimizer.n_sample)+'.pth')
                # grads_path = os.path.join(configs.GraSP.summary_dir + 'N_'+str(configs.optimizer.n_sample)+'.pth')

            # if isinstance(optimizer, ZO_SCD_mask):
            #     grads_path = os.path.join('./figs/' + configs.GraSP.network + configs.optimizer.name + '/h_'+str(configs.optimizer.h_smooth)+'.pth')
            # elif isinstance(optimizer, ZO_SGD_mask):
            #     grads_path = os.path.join('./figs/' + configs.GraSP.network + configs.optimizer.name + '/N_'+str(configs.optimizer.n_sample)+'.pth')

            torch.save(grads, grads_path)

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
    
    # ================== scheduler ======================
    lr_schedulers = build_scheduler(config, optimizers, learning_rate)
    
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
                optimizer = optimizers[1]
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
                'masks': mb.masks,
                'named_masks': named_masks,
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

    set_torch_deterministic(42)

    # ================== Prepare logger ==========================
    name_append = 'bz{}_grad{}_lr_{}'.format(configs.GraSP.batch_size, configs.optimizer.grad_sparsity, configs.GraSP.learning_rate)
    paths = [configs.GraSP.dataset]
    summn = [configs.GraSP.network, configs.optimizer.name, str(configs.GraSP.pruner), configs.GraSP.exp_name, name_append, time.strftime("%Y%m%d-%H%M%S")]
    chekn = [configs.GraSP.network, configs.optimizer.name, str(configs.GraSP.pruner), configs.GraSP.exp_name, name_append, time.strftime("%Y%m%d-%H%M%S")]
    if configs.run.runs is not None:
        summn.append('run_%s/' % configs.run.runs)
        chekn.append('run_%s/' % configs.run.runs)
    # summn.append("summary/")
    # chekn.append("checkpoint/")
    summary_dir = ["./runs/pruning"] + paths + summn
    ckpt_dir = ["./runs/pruning"] + paths + chekn
    # summary_dir, ckpt_dir is path (with / in the end)
    configs.GraSP.summary_dir = os.path.join(*summary_dir)
    configs.GraSP.checkpoint_dir = os.path.join(*ckpt_dir)
    print("=> config.summary_dir:    %s" % configs.GraSP.summary_dir)
    print("=> config.checkpoint_dir: %s" % configs.GraSP.checkpoint_dir)

    # save .yml to directory
    makedirs(configs.GraSP.summary_dir)
    makedirs(configs.GraSP.checkpoint_dir)
    shutil.copy(args.config, configs.GraSP.summary_dir)
    
    logger, writer = init_logger(configs.GraSP)
    # logger.info(dict(configs))
    logger.info(dict(configs.GraSP))
    logger.info(dict(configs.optimizer))
    logger.info(dict(configs.scheduler))
    logger.info(dict(configs.model))
    logger.info(os.getpid())
    
    # ====================================== graph and stat ======================================
    # t_batch = next(iter(training_data))
    # targets, inputs, slot_label,attn,seg = map(lambda x: x.to(device), t_batch)
    # writer.add_graph(model, (inputs,attn,seg))
    # writer.close()
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())

    # preprocessing
    # ====================================== get dataloader ======================================
    trainloader, testloader = get_dataloader(configs.GraSP.dataset, configs.GraSP.batch_size, 256, 4)
    
    # ====================================== fetch configs ======================================
    ckpt_path = configs.GraSP.checkpoint_dir
    num_iterations = configs.GraSP.iterations
    target_ratio = configs.GraSP.target_ratio
    normalize = configs.GraSP.normalize

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

    # ====================================== Prepare model ======================================
    iteration = 0
    
    # has pretrained model
    if hasattr(configs, 'pretrained') and configs.pretrained.incre == True:
        model_state = torch.load(configs.pretrained.load_model_path)
        model = model_state['net']
        logger.info('Pre-trained model accuracy: %.4f ' % model_state['acc'])
    # from scratch
    else: 
        model = build_model(configs)
    
    # build ModelBase 
    mb = ModelBase(configs.GraSP.network, configs.GraSP.depth, configs.GraSP.dataset, model)
    mb.cuda()

    # ====================================== Pruning Masks ======================================
    # @ToDo: check whether this attribute exists
    # if hasattr(configs, 'pretrained') and hasattr(configs.pretrained, 'pretrained'):
    
    # has pretrained pruned masks:
    if hasattr(configs, 'pretrained') and configs.pretrained.incre == True and configs.pretrained.pruned == True:
        masks = model_state['masks']
        named_masks = model_state['named_masks']
        # keep pruned params 0
        mb.register_mask(masks, forward_hook=True)
    # generate mask this time
    elif configs.GraSP.pruner != False:
        # ================== start pruning ==================
        
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    1,
                                                                                    num_iterations))

        if hasattr(configs, 'pretrained') and configs.pretrained.incre == True:
            pass
        else:
            mb.model.apply(weights_init)
            print("=> Applying weight initialization(%s)." % configs.GraSP.get('init_method', 'kaiming'))
        
        print("Iteration of: %d/%d" % (iteration, num_iterations))

        masks, named_masks = GraSP_zo_mask(mb.model, ratio, trainloader, 'cuda',
                      num_classes=configs.GraSP.num_classes,
                      samples_per_class=configs.GraSP.samples_per_class,
                      num_iters=configs.GraSP.num_iters,
                      tensorized=configs.model.tensorized)
        iteration = 0
        print('=> Using GraSP')
        # ========== register mask ==================
        # pretrained model, do not register forward hook (remain original value)
        if hasattr(configs, 'pretrained') and configs.pretrained.incre == True:
            mb.register_mask(masks, forward_hook=False)
        # from scratch, register forward hook (keep pruned params as 0)
        else:
            mb.register_mask(masks, forward_hook=True)
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
        masks = None
        named_masks = None
    # ======================= Training =======================
    logger.info(configs.GraSP.exp_name)
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
