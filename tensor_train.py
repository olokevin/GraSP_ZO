from random import shuffle
from re import T
import tabnanny
import torch
import torch.nn as nn
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import QNLI
from datasets import load_dataset
from typing import Iterable, List
import argparse
import math
import time
import os
import shutil

from tqdm import tqdm
import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
# from dataset import TranslationDataset, paired_collate_fn
from torch.utils.data import Dataset

import numpy as np

# from sklearn import metrics
import sklearn.metrics

from tensorboardX import SummaryWriter
from torchstat import stat
from tqdm import tqdm
from utils.init_utils import weights_init
from utils.common_utils import (get_logger, makedirs, process_config, PresetLRScheduler, str_to_list)
from utils.builder import build_optimizer, build_scheduler
# from utils.data_utils import get_dataloader
# from utils.network_utils import get_network
from models.model_base import ModelBase
from pruner.GraSP_attn import GraSP_attn

from pyutils.config import configs

from optimizer import ZO_SCD_mask, ZO_SGD_mask

torch.manual_seed(20211214)
# torch.manual_seed(0)

# torch.manual_seed(20211213)


# We need to modify the URLs for the dataset since the links to the original dataset are broken
# Refer to https://github.com/pytorch/text/issues/1756#issuecomment-1163664163 for more info

# torch.backends.cudnn.enabled = True
# torch.backends.cudnn.benchmark = True
class CustomDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return self.data[idx]


def get_kl_loss(model, epoch, tensor_blocks = None, no_kl_epochs=5, warmup_epochs=10):

    kl_loss = 0.0

    for layer in tensor_blocks:
        kl_loss += layer.tensor.compute_loglikelihood()

    kl_mult = 1e-4 * torch.clamp(
                            torch.tensor((
                                (epoch - no_kl_epochs) / warmup_epochs)), 0.0, 1.0)
    """
    print("KL loss ",kl_loss.item())
    print("KL Mult ",kl_mult.item())
    """
    return kl_loss*kl_mult

SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}



from torch.nn.utils.rnn import pad_sequence

def collate_fn_custom(batch):
    src_batch, sim_batch = [], []
    src_attn_batch = []
    slot_batch = []
    seg_batch = []
    for similar,seq,slot,seg in batch:
        
        seq = np.insert(seq,0,0)
        seg.append(1)
        src_batch.append(torch.tensor(seq))
        src_attn_batch.append(torch.tensor([1]*len(src_batch[-1])))
        # tgt_batch.append(text_transform(tgt_sample.rstrip("\n")))
        sim_batch.append(similar)
        # seg_batch.append(torch.tensor(seg))
        seg_batch.append(torch.tensor([1]*len(src_batch[-1])))
        slot_batch.append(torch.tensor(slot)-1) #ATIS 
        # slot_batch.append(torch.tensor(slot)) #en


        # sim_batch.append(similar-1)
        # seg_batch.append(torch.tensor(seg))
        # slot_batch.append(torch.tensor(slot)-2)

        # print(slot)
    src_batch = pad_sequence(src_batch, padding_value=0)
    src_attn_batch = pad_sequence(src_attn_batch, padding_value=0)
    seg_batch = pad_sequence(seg_batch, padding_value=0)
    slot_batch = pad_sequence(slot_batch, padding_value=-100)
    # tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)

    return torch.tensor(sim_batch),torch.swapaxes(src_batch,0,1),torch.swapaxes(slot_batch,0,1),torch.swapaxes(src_attn_batch,0,1),torch.swapaxes(seg_batch,0,1)

# ==================== GraSP methods ====================
def init_logger(config):
    makedirs(config.summary_dir)
    makedirs(config.checkpoint_dir)

    # set logger
    path = os.path.dirname(os.path.abspath(__file__))
    path_model = os.path.join(path, 'tensor_layers/Transformer_tensor/%s.py' % config.network.lower())
    path_main = os.path.join(path, 'tensor_train.py')
    path_pruner = os.path.join(path, 'pruner/%s.py' % config.pruner_file)
    logger = get_logger('log', logpath=config.summary_dir + '/',
                        filepath=path_model, package_files=[path_main, path_pruner], displaying=False)
    logger.info(dict(config))
    writer = SummaryWriter(config.summary_dir)
    # sys.stdout = open(os.path.join(config.summary_dir, 'stdout.txt'), 'w+')
    # sys.stderr = open(os.path.join(config.summary_dir, 'stderr.txt'), 'w+')
    return logger, writer

def get_exception_layers(net, exception):
    exc = []
    idx = 0
    for m in net.modules():
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            if idx in exception:
                exc.append(m)
            idx += 1
    return tuple(exc)

def print_mask_information(mb, logger):
    ratios = mb.get_ratio_at_each_layer()
    logger.info('** Mask information of %s. Overall Remaining: %.2f%%' % (mb.get_name(), ratios['ratio']))
    count = 0
    for k, v in ratios.items():
        if k == 'ratio':
            continue
        logger.info('  (%d) %s: Remaining: %.2f%%' % (count, k, v))
        count += 1

def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {'Total': total_num, 'Trainable': trainable_num}

''' Main function '''
parser = argparse.ArgumentParser()
parser.add_argument('-ops_idx')

parser.add_argument('-tensorized', type=int, default=1)
parser.add_argument('-quantized', type=int, default=1)
parser.add_argument('-uncompressed', type=int, default=0)
parser.add_argument('-precondition', type=int, default=0)

parser.add_argument('-epoch', type=int, default=100)
parser.add_argument('-batch_size', type=int, default=64)

#parser.add_argument('-d_word_vec', type=int, default=512)
parser.add_argument('-d_model', type=int, default=768)
parser.add_argument('-d_inner_hid', type=int, default=2048)
parser.add_argument('-d_k', type=int, default=64)
parser.add_argument('-d_v', type=int, default=64)

parser.add_argument('-n_head', type=int, default=8)
parser.add_argument('-n_layers', type=int, default=6)
parser.add_argument('-n_warmup_steps', type=int, default=40000)

parser.add_argument('-dropout', type=float, default=0.1)
parser.add_argument('-embs_share_weight', action='store_true')
parser.add_argument('-proj_share_weight', action='store_true')

parser.add_argument('-log', default=None)
parser.add_argument('-save_model', default=None)
parser.add_argument('-save_mode', type=str, choices=['all', 'best'], default='best')

parser.add_argument('-no_cuda', action='store_true')
parser.add_argument('-label_smoothing', action='store_true')

parser.add_argument('-config', metavar='FILE', help='config file')

opt = parser.parse_args()

#========= Loading Dataset =========#
from torch.utils.data import DataLoader

train_iter = torch.load('processed_data/ATIS_train.pt')

training_data = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=True)

val_iter = torch.load('processed_data/ATIS_valid.pt')

validation_data = DataLoader(val_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=False)

test_iter = torch.load('processed_data/ATIS_test.pt')

test_data = DataLoader(test_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=False)

opt.src_vocab_size = 19213
opt.tgt_vocab_size = 10839

# ================== Preparing Model ================== #
if opt.embs_share_weight:
    assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
        'The src/tgt word2idx table are different but asked to share word embedding.'

print(opt)

device = torch.device('cuda')

# ========= Model Shape ========= #

n_src_vocab = 148669 ##QNLI
n_src_vocab = 143179 ##MNLI

# n_src_vocab = 30522 ## BERT TOKEN 
n_src_vocab = 800 ## ATIS TOKEN 
# n_src_vocab = 6329 ## EN TOKEN 

n_trg_vocab = 10839
d_word_vec = 768
n_layers = 2
n_head = 12
d_q = d_word_vec//n_head
d_k = d_word_vec//n_head
d_v = d_word_vec//n_head
d_model = 768
d_inner = 3072
pad_idx = None
dropout = 0.1
n_position = 512
scale_emb = True

# ========= Tensor Shape ========= #

# emb_shape = [[5,5,4,4,2],[3,4,4,4,4]] #ATIS shape
emb_shape = [[8,5,5,4,8],[4,4,4,4,3]]

emb_rank = 30
emb_tensor_type = 'TensorTrainMatrix'

trg_shape = [[10,10,10,11],[4,4,8,6]]
trg_emb_rank = 30
trg_tensor_type = 'TensorTrainMatrix'

r = 20
r_attn = 20

# ========= Yequan's TensorTrainMatrix Version ========= #
# attention_shape = [[[12,8,8],[8,8,12]], [[12,8,8],[8,8,12]]]
# attention_rank = [[1,r_attn,r_attn,1], [1,r_attn,r_attn,1]]
# attention_tensor_type = 'TensorTrainMatrix'

# ffn_shape = [[[12,8,8],[12,16,16]], [[16,16,12],[8,8,12]]]
# ffn_rank = [[1,r,r,1], [1,r,r,1]]

# ffn_tensor_type = 'TensorTrainMatrix'

# classifier_shape = [[12,8,8], [8,8,12]]
# classifier_rank = [1,r,r,1]
# classifier_tensor_type = 'TensorTrainMatrix'

# ========= Zi Yang's TensorTrain Version ========= #
attention_shape = [[12,8,8,8,8,12],[12,8,8,8,8,12]]
attention_rank = [[1,12,r_attn,r_attn,r_attn,12,1], [1,12,r_attn,r_attn,r_attn,12,1]]
attention_tensor_type = 'TensorTrain'

ffn_shape = [[12,8,8,12,16,16],[16,16,12,8,8,12]]
ffn_rank = [[1,12,r,r,r,16,1], [1,16,r,r,r,12,1]]
ffn_tensor_type = 'TensorTrain'

classifier_shape = [12,8,8,8,8,12]
classifier_rank = [1,12,r,r,r,12,1]
classifier_tensor_type = 'TensorTrain'

# ========= Quant ========= #

bit_attn = 8
scale_attn = 2**(-5)
bit_ffn = 8
scale_ffn = 2**(-5)
bit_a = 8
scale_a = 2**(-5)


d_classifier=768
num_class = 22 #ATIS
slot_num = 121 #ATIS

# num_class = 60 #en
# slot_num = 56 #en
dropout_classifier = 0.1

quantized = False
tensorized = (opt.tensorized == 1)
precondition = False

uncompressed = opt.uncompressed


from tensor_layers.Transformer_tensor import Transformer_sentence_concat_SLU
transformer = Transformer_sentence_concat_SLU(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            d_classifier=d_classifier,num_class=num_class,dropout_classifier=dropout_classifier,
            classifier_shape = classifier_shape,classifier_rank = classifier_rank,classifier_tensor_type = classifier_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            slot_num = slot_num,
            quantized=quantized,
            tensorized = tensorized,
            uncompressed=uncompressed)

print(transformer)

from tensor_layers.layers import TensorizedLinear_module

transformer_old = Transformer_sentence_concat_SLU(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            d_classifier=d_classifier,num_class=num_class,dropout_classifier=dropout_classifier,
            classifier_shape = classifier_shape,classifier_rank = classifier_rank,classifier_tensor_type = classifier_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            slot_num = slot_num,
            quantized=quantized,
            tensorized = tensorized,
            uncompressed=uncompressed)


transformer.to(device)
transformer_old.to(device)

precondition = False

model = transformer

tensor_blocks = []
layer_notensor = []

layer_notensor = nn.ModuleList([model._modules['encoder']._modules['src_word_emb'], model._modules['classifier'][-1],model._modules['encoder'].token_type_emb,model._modules['encoder'].position_enc,model._modules['encoder'].layer_norm])

for block in model._modules['encoder']._modules['layer_stack']:
    attn = block.slf_attn
    ppn = block.pos_ffn

    for mods in attn._modules.values():
        if hasattr(mods, 'tensor'):
            tensor_blocks.append(mods)
        else:
            layer_notensor.append(mods)
    for mods in ppn._modules.values():
        if hasattr(mods, 'tensor'):
            tensor_blocks.append(mods)
        else:
            layer_notensor.append(mods)
tensor_blocks.append(model._modules['classifier'][0])
tensor_blocks = nn.ModuleList(tensor_blocks)

tensor_blocks.append(model.slot_classifier[0])


layer_notensor.append(model.slot_classifier[2])
layer_notensor.append(model.slot_classifier[-1])


#========= Preparing GraSP =========#
def main():

    PATH = 'model/test_ATIS_tensor_2layer_INT4.chkpt'
    PATH = 'model/test.chkpt'
    PATH = 'model/ATIS_tensor_2layers_FP8_input52_factor52_med52_grad52_dynamic_1e-3_test.chkpt'
    PATH = 'model/ATIS_tensor_2layers_FP8_all52_dynamic_1e-3_new.chkpt'
    PATH = 'model/test_ZO.chkpt'
    # PATH = 'models_end2end/MNLI_r20_nopre.chkpt'
    # transformer.load_state_dict(torch.load(PATH))

    epochs = 200

    # ================== .yml parser ==========================
    # Config: path of *.yml configuration file
    # Currently disabled, and added at front
    
    # parser = argparse.ArgumentParser()
    # parser.add_argument('config', metavar='FILE', help='config file')
    # args, opts = parser.parse_known_args()

    # load config *.yml file
    # recursive: also load default.yaml
    configs.load(opt.config, recursive=False)
    
    # ================== Prepare logger ==========================
    paths = [configs.GraSP.dataset]
    summn = [configs.GraSP.network, configs.optimizer.name, configs.GraSP.exp_name, time.strftime("%Y%m%d-%H%M%S")]
    chekn = [configs.GraSP.network, configs.optimizer.name, configs.GraSP.exp_name, time.strftime("%Y%m%d-%H%M%S")]
    if configs.run.runs is not None:
        summn.append('run_%s' % configs.run.runs)
        chekn.append('run_%s' % configs.run.runs)
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
    shutil.copy(opt.config, configs.GraSP.summary_dir)

    logger, writer = init_logger(configs.GraSP)
    logger.info(dict(configs))

    # ====================================== graph and stat ======================================
    # t_batch = next(iter(training_data))
    # targets, inputs, slot_label,attn,seg = map(lambda x: x.to(device), t_batch)
    
    # writer.add_graph(model, (inputs,attn,seg))
    # writer.close()
    # for name,parameters in model.named_parameters():
    #     print(name,':',parameters.size())


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
    learning_rates = float(configs.GraSP.learning_rate)
    weight_decays = float(configs.GraSP.weight_decay)
    training_epochs = int(configs.GraSP.epoch)
    logger.info('Normalize: %s, Total iteration: %d, Target ratio: %.2f, Iter ratio %.4f.' %
                (normalize, num_iterations, target_ratio, ratio))

    # ====================================== start pruning ======================================
    # we need: model, mb, masks, named_masks
    iteration = 0

    # has pretrained model
    if hasattr(configs, 'pretrained') and configs.pretrained.incre == True:
        model_state = torch.load(configs.pretrained.load_model_path)
        model = model_state['net']
        logger.info('Pre-trained model accuracy: %.4f ' % model_state['acc'])
    # from scratch
    else: 
        model = transformer
    
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
    # Need pruning
    elif configs.GraSP.pruner != False:
        logger.info('** Target ratio: %.4f, iter ratio: %.4f, iteration: %d/%d.' % (target_ratio,
                                                                                    ratio,
                                                                                    1,
                                                                                    num_iterations))

        # ====================================== build ModelBase ====================================== 
        model = transformer
        mb = ModelBase(configs.GraSP.network, configs.GraSP.depth, configs.GraSP.dataset, model)
        mb.cuda()
        mb.model.apply(weights_init)
        print("=> Applying weight initialization(%s)." % configs.GraSP.get('init_method', 'kaiming'))
        print("Iteration of: %d/%d" % (iteration, num_iterations))

        sample_dataloader = DataLoader(train_iter, batch_size=configs.GraSP.samples_batches*configs.run.batch_size, collate_fn=collate_fn_custom, shuffle=True)

        masks, named_masks = GraSP_attn(mb.model, ratio, sample_dataloader, 'cuda',
                      num_iters=configs.GraSP.num_iters)
                      # samples_batches = configs.GraSP.samples_batches,
                      # num_classes=configs.GraSP.num_classes,
                      # samples_per_class=configs.GraSP.samples_per_class,
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
        masks = None
        named_masks = None
    
    print(get_parameter_number(model))
        
    
    # valid_loss, valid_accu, valid_slot_accu = eval_epoch(transformer, validation_data, device, weight=None)
    block = transformer.encoder.layer_stack[0].slf_attn.w_qs

    precondition = (opt.precondition==1)
    
    step = 2

    # transformer.to(torch.bfloat16)
    par = list(transformer.parameters())
    # for p in par:
    #     p.requires_grad = True
    
    # lr = 1e-3

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    if configs.optimizer.name == 'ZO_SCD_mask':
        optimizer_notensor = ZO_SCD_mask(
                model = model, 
                criterion = criterion,
                masks = named_masks,
                lr = learning_rates,
                grad_sparsity = configs.optimizer.grad_sparsity,
                tensorized = 'None'
            )
        optimizer_tensor = ZO_SCD_mask(
                model = model, 
                criterion = criterion,
                masks = named_masks,
                lr = learning_rates,
                grad_sparsity = configs.optimizer.grad_sparsity,
                tensorized = configs.model.tensorized
            )
    elif configs.optimizer.name == 'ZO_SGD_mask':
        optimizer_notensor = ZO_SGD_mask(
            model = model, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rates,
            sigma = configs.optimizer.sigma,
            n_sample  = configs.optimizer.n_sample,
            signSGD = configs.optimizer.signSGD,
            layer_by_layer = configs.optimizer.layer_by_layer,
            tensorized = 'None'
        )
        optimizer_tensor = ZO_SGD_mask(
            model = model, 
            criterion = criterion,
            masks = named_masks,
            lr = learning_rates,
            sigma = configs.optimizer.sigma,
            n_sample  = configs.optimizer.n_sample,
            signSGD = configs.optimizer.signSGD,
            layer_by_layer = configs.optimizer.layer_by_layer,
            tensorized = configs.model.tensorized
        )
    elif  configs.optimizer.name == 'ADAM':
        optimizer_tensor = optim.Adam(filter(lambda x: x.requires_grad, tensor_blocks.parameters()),betas=(0.9, 0.98), eps=1e-06, lr = learning_rates)
        optimizer_notensor = optim.Adam(layer_notensor.parameters(), betas=(0.9, 0.98), eps=1e-06, lr = learning_rates)
    else:
        raise ValueError(f"Wrong optimizer_name {configs.optimizer.name}") 

    
    # optimizers = build_optimizer(configs, model, criterion, named_masks, learning_rates, weight_decays)
    lr_scheduler = build_scheduler(configs, optimizer_tensor, learning_rates)

    # ===================== Zi Yang's optimzier setting =====================
    
    # if precondition == True:
    #     optimizer_tensor = torch.optim.SGD(tensor_blocks.parameters(),lr = 1e-1)
    #     optimizer = optim.Adam(filter(lambda x: x.requires_grad, layer_notensor.parameters()),
    #                 betas=(0.9, 0.98), eps=1e-09, lr = 1e-3)
    # else:
    #     if tensorized == True:
    #         lr = 1e-3
    #         # optimizer_tensor = torch.optim.SGD(tensor_blocks.parameters(),lr = 1e-1)
    #         optimizer_tensor = optim.Adam(filter(lambda x: x.requires_grad, tensor_blocks.parameters()),betas=(0.9, 0.98), eps=1e-06, lr = lr)
    #         optimizer = optim.Adam(layer_notensor.parameters(), betas=(0.9, 0.98), eps=1e-06, lr = lr)

    #         optimizer_ZO = None

    #     else:
    #         lr = 1e-3
    #         optimizer = optim.Adam(
    #                         filter(lambda x: x.requires_grad, transformer.parameters()),
    #                         betas=(0.9, 0.98), eps=1e-06, lr = lr)
    #         optimizer_tensor = None
            
    #         transformer.requires_grad_(False)
    #         optimizer_ZO = None
    
    valid_acc_all = [-1]
    train_result = []
    test_result = []
    for epoch in range(training_epochs):
        # ========================== Setup Optimizer ==========================
        # # mix training selection
        # if configs.optimizer.name == 'ZO_mix':
        #     if epoch < configs.optimizer.switch_epoch:
        #         optimizer = optimizers[0]
        #         lr_scheduler = lr_schedulers[0]
        #     else:
        #         optimizer = optimizers[1]
        #         lr_scheduler = lr_schedulers[1]
        # # single training
        # else:
        #     optimizer = optimizers
        #     lr_scheduler = lr_schedulers
        
        start = time.time()

        # ========================== Training ==========================
        # if isinstance(lr_scheduler, PresetLRScheduler):
        #     lr_scheduler(optimizer, epoch)
        #     now_lr = lr_scheduler.get_lr(optimizer)
        # else:
        #     now_lr = optimizer.state_dict()['param_groups'][0]['lr']
        
        if isinstance(lr_scheduler, PresetLRScheduler):
            lr_scheduler(optimizer_notensor, epoch)
            lr_scheduler(optimizer_tensor, epoch)
            now_lr = lr_scheduler.get_lr(optimizer_tensor)
        else:
            pass
        
        train_loss, train_accu, train_slot_accu = train_epoch_bylayer(
            transformer, training_data,
            Loss=criterion,
            optimizer_notensor=optimizer_notensor,
            optimizer_tensor=optimizer_tensor,
            precondition=precondition,device=device,tensor_blocks=tensor_blocks,
            step=epoch)

        # train_loss, train_accu = 0,0

        start_val = time.time()

        # ========================== Validation ==========================
        valid_loss, valid_accu, valid_slot_accu = eval_epoch(transformer, validation_data, device, weight=None)

        # ========================== Test ==========================
        test_loss, test_accu, test_slot_accu = eval_epoch(transformer, test_data, device, weight=None)
        # train_loss_new, train_accu = eval_epoch(transformer, training_data, device, weight=None)
        train_result += [train_loss.cpu().to(torch.float32),train_accu.cpu().to(torch.float32),train_slot_accu.cpu().to(torch.float32)]
        test_result += [test_loss,test_accu.cpu().to(torch.float32),test_slot_accu]

        if isinstance(lr_scheduler, PresetLRScheduler):
            pass
        else:
            lr_scheduler.step()

        end = time.time()

        train_loss_new = 0
        
        print('')
        print('epoch = ', epoch)
        
        print('  - (Training)   loss: {loss: 8.5f}, loss_hard: {loss_new: 8.5f}, accuracy: {accu:3.3f} %, slot accuracy: {slot_accu:3.3f},'\
            'elapse: {elapse:3.3f} min'.format(
                loss=train_loss, loss_new=train_loss_new, accu=100*train_accu, slot_accu = 100*train_slot_accu,
                elapse=(start_val-start)/60))
       
        
        print('  - (Validation) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, slot accuracy: {slot_accu:3.3f},'\
                'elapse: {elapse:3.3f} min'.format(
                    loss=valid_loss, accu=100*valid_accu,slot_accu = 100*valid_slot_accu,
                    elapse=(end-start_val)/60))

        print('  - (Test) loss: {loss: 8.5f}, accuracy: {accu:3.3f} %, slot accuracy: {slot_accu:3.3f},'\
                'elapse: {elapse:3.3f} min'.format(
                    loss=test_loss, accu=100*test_accu,slot_accu = 100*test_slot_accu,
                    elapse=(end-start_val)/60))
        
        # full_model_name = './model/' + configs.GraSP.exp_name + '.chkpt'
        # torch.save(transformer.state_dict(),best_model_name)
        # torch.save(transformer,full_model_name)

        if max(valid_acc_all)<valid_accu:
            best_model_name = './model/' + configs.GraSP.exp_name + '_best' + '.chkpt'
            # torch.save(transformer.state_dict(),best_model_name)
            if configs.GraSP.pruner != False:
                result = {
                    'model': model,
                    'masks': masks,
                    'named_masks': named_masks
                }
                torch.save(result,best_model_name)
            else:
                torch.save(model,best_model_name)
        valid_acc_all.append(valid_accu)
    PATH_np = './model/' + configs.GraSP.exp_name + '.npy'
    np.save(PATH_np,np.array([train_result,test_result]))

def train_epoch_bylayer(model, training_data, Loss, optimizer_notensor, optimizer_tensor, tensor_blocks=None,precondition=False,device='cuda',step=1):
    ''' Epoch operation in training phase'''

    model.train()
    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    cos_total = 0
    attn_total = 0

    slot_total = 0
    slot_correct = 0

    count = 0

    max_memory = 0

    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        target, w1, slot_label,attn,seg= map(lambda x: x.to(device), batch)

        optimizer_tensor.zero_grad()
        optimizer_notensor.zero_grad()

        # attn = None
        # model.eval()
        
        memory = torch.cuda.max_memory_allocated()/1024/1024/1024
        max_memory = max(max_memory,memory)

        inputs = (w1, attn, seg)
        targets = (target, slot_label)

        # Forward
        pred,pred_slot = model(w1,attn=attn,seg=seg)
        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)
        slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)

        loss_MLM =  Loss(pred_slot, slot_label)
        loss = Loss(pred,target)  + loss_MLM

        # FO / ZO
        if isinstance(optimizer_tensor, ZO_SCD_mask):
            optimizer_tensor.step(inputs, targets, ATIS=True)
        elif isinstance(optimizer_tensor, ZO_SGD_mask):
            optimizer_tensor.step(inputs, targets, ATIS=True)
        else:
            loss.backward()
            optimizer_tensor.step()
        
        if isinstance(optimizer_notensor, ZO_SCD_mask):
            optimizer_notensor.step(inputs, targets, ATIS=True)
        elif isinstance(optimizer_notensor, ZO_SGD_mask):
            optimizer_notensor.step(inputs, targets, ATIS=True)
        else:
            optimizer_notensor.step()
            
        # print('ZO=',model.encoder.layer_stack[-1].slf_attn.w_ks.tensor.factors[2].grad[1,:10,1])

        # optimizer.zero_grad()
        # optimizer_tensor.zero_grad()
        # print('Zero=',model.encoder.layer_stack[-1].slf_attn.w_ks.tensor.factors[2].grad[1,:10,1])
        # loss.backward()
        # print('grad=',model.encoder.layer_stack[-1].slf_attn.w_ks.tensor.factors[2].grad[1,:10,1])


        # torch.nn.utils.clip_grad_norm_(tensor_blocks.parameters(), 0.25)
        if precondition==True:
            device = 'cuda' 
    
            for U in tensor_blocks:
                T = U.tensor
                left = torch.squeeze(T.factors[0])
                for V in T.factors[1:]:
                    tmp = left.T@left
                    r = tmp.shape[0]
                    V.grad = torch.tensordot(torch.inverse(tmp+0.001*torch.eye(r).to(device)), V.grad,[[-1],[0]]).clone()
                    left = torch.flatten(torch.tensordot(left,V,[[-1],[0]]),start_dim=0,end_dim=1)

                right = torch.squeeze(T.factors[-1])
                for V in T.factors[::-1][1:]:
                    tmp = right@right.T
                    r = tmp.shape[0]
                    V.grad =  torch.tensordot(V.grad,torch.inverse(tmp+0.001*torch.eye(r).to(device)),[[-1],[0]]).clone()
                    right = torch.flatten(torch.tensordot(V,right,[[-1],[0]]),start_dim=1,end_dim=-1)
                    
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        

        # note keeping
        total_loss += loss.detach()
        n_word_total += pred.shape[0]
        n_word_correct += torch.sum(torch.argmax(pred.detach(),dim=1)==target)

        slot_total += torch.sum(slot_label>=0).detach()
        slot_correct += torch.sum((torch.argmax(pred_slot.detach(),dim=-1)==slot_label)*(slot_label>=0)).detach()

        # print(torch.argmax(pred.detach(),dim=1))



        # print(torch.argmax(pred,dim=1)[:10])
        # print(target[:10])
        # print(pred[:10,:])
        count += 1
        
        # if count == 2:
        #     break
        


        if count%500==0:
            print('loss = ', total_loss/n_word_total)
            print('acc = ', n_word_correct/n_word_total)
            print("torch.cuda.memory_allocated: %fGB"%(max_memory))
            # for p in model.parameters():
            #     if p.grad!=None:
            #         print(p[0])
            #         print('grad=', p.grad[0])
            #         break
        # if count==10:
        #     break

    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    accuracy_slot = slot_correct/slot_total

    return loss_per_word, accuracy, accuracy_slot



def eval_epoch(model, validation_data, device, **kwargs):
    ''' Epoch operation in evaluation phase '''

    model.eval()

    total_loss = 0
    n_word_total = 0
    n_word_correct = 0

    slot_total = 0
    slot_correct = 0

    slot_label_total = torch.tensor([0])
    pred_label_total = torch.tensor([0])

    slot_correct_all = []
    slot_pred_all = []

    Loss = nn.CrossEntropyLoss(label_smoothing=0.1,ignore_index=- 100)

    with torch.no_grad():
        for batch in tqdm(
                validation_data, mininterval=2,
                desc='  - (Validation) ', leave=False):

            target, w1, slot_label,attn,seg= map(lambda x: x.to(device), batch)


            # attn = None
            pred,pred_slot = model(w1,attn=attn,seg=seg)
            pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)

            slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)

            # print(pred_slot.shape)
            # print(slot_label[(slot_label<0) * (slot_label!=-100)])
            # # slot_label = torch.max(torch.tensor(0),slot_label)
            # print('max=',torch.max(slot_label))
            # print('min=',torch.min(slot_label[slot_label!=-100]))


            # print(pred_slot.shape)
            loss_MLM =  Loss(pred_slot, slot_label)

            #     # pred_un1 = transformer_un1(w1,attn)


            # # loss = Loss(pred,target) 
            loss = loss_MLM +  Loss(pred,target) 


        # print(loss)
            total_loss += loss.item()



            n_word_total += pred.shape[0]
            n_word_correct += torch.sum(torch.argmax(pred,dim=1)==target)

            slot_total += torch.sum(slot_label>=0).detach()
            slot_correct += torch.sum((torch.argmax(pred_slot.detach(),dim=-1)==slot_label)*(slot_label>=0)).detach()

            pred_label = (torch.argmax(pred_slot.detach(),dim=-1))[slot_label>=0]
            # pred_slot[slot_label>=0]
            slot_label = slot_label[slot_label>=0]

            # print(slot_label.shape)
            # print(pred_label.shape)


            slot_label_total=torch.cat((slot_label_total, slot_label.to('cpu')))
            pred_label_total=torch.cat((pred_label_total, pred_label.to('cpu')))

            slot_correct_all.append(slot_label.tolist())
            slot_pred_all.append(pred_label.tolist())



            
    # print(slot_label_total.shape)
    # print(pred_label_total.shape)

    # f1_score = sklearn.metrics.f1_score(slot_label_total[1:].numpy(),pred_label_total[1:].numpy(),average='weighted')
    f1_score = sklearn.metrics.f1_score(slot_label_total[1:].numpy(),pred_label_total[1:].numpy(),average='micro') # old is average = 'weighted'



    loss_per_word = total_loss/n_word_total
    accuracy = n_word_correct/n_word_total
    accuracy_slot = slot_correct/slot_total
    # accuracy_slot
    return loss_per_word, accuracy, f1_score


def compute_F1(pred,target):
    pass

if __name__ == '__main__':
    print(len(vocab_transform))
    # print(len(vocab_transform[TGT_LANGUAGE]))
    main()
