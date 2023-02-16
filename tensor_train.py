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

opt = parser.parse_args()

#========= Loading Dataset =========#
# data = torch.load(opt.data)
# opt.max_token_seq_len = data['settings'].max_token_seq_len

from torch.utils.data import DataLoader

# training_data, validation_data = prepare_dataloaders(data, opt)
train_iter = torch.load('processed_data/ATIS_train.pt')
# train_iter = torch.load('processed_data/snips_train.pt')
# train_iter = torch.load('processed_data/en_train.pt')


# train_iter = load_dataset("glue",'mnli',split='train')
training_data = DataLoader(train_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=True)

# val_iter = load_dataset("glue",'mnli',split='validation_matched')
val_iter = torch.load('processed_data/ATIS_valid.pt')
# val_iter = torch.load('processed_data/snips_valid.pt')
# val_iter = torch.load('processed_data/en_valid.pt')

validation_data = DataLoader(val_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=False)

test_iter = torch.load('processed_data/ATIS_test.pt')
# test_iter = torch.load('processed_data/snips_test.pt')
# test_iter = torch.load('processed_data/en_test.pt')

test_data = DataLoader(test_iter, batch_size=opt.batch_size, collate_fn=collate_fn_custom, shuffle=False)

opt.src_vocab_size = 19213
opt.tgt_vocab_size = 10839

#========= Preparing Model =========#
if opt.embs_share_weight:
    assert training_data.dataset.src_word2idx == training_data.dataset.tgt_word2idx, \
        'The src/tgt word2idx table are different but asked to share word embedding.'

print(opt)

device = torch.device('cuda')

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

# emb_shape = [[15,20,20,25],[4,4,8,6]]
# emb_shape = [[16,20,10,10],[4,4,8,6]]
# emb_shape = [[5,5,4,8],[4,4,8,6]]
# emb_shape = [[5,5,4,4,2],[3,4,4,4,4]] #ATIS shape
emb_shape = [[8,5,5,4,8],[4,4,4,4,3]]
# emb_shape = [[10,10,10,10],[4,4,8,6]]


emb_rank = 30
emb_tensor_type = 'TensorTrainMatrix'

trg_shape = [[10,10,10,11],[4,4,8,6]]
trg_emb_rank = 30
trg_tensor_type = 'TensorTrainMatrix'

r = 20
r_attn = 20

attention_shape = [[12,8,8,8,8,12],[12,8,8,8,8,12]]
# attention_rank = [r_attn,r_attn]
attention_rank = [[1,12,r_attn,r_attn,r_attn,12,1], [1,12,r_attn,r_attn,r_attn,12,1]]
# attention_rank = [[1,r_attn,r_attn,r_attn,r_attn,r_attn,1], [1,r_attn,r_attn,r_attn,r_attn,r_attn,1]]
attention_tensor_type = 'TensorTrain'

ffn_shape = [[12,8,8,12,16,16],[16,16,12,8,8,12]]
# ffn_rank = [r,r]
ffn_rank = [[1,12,r,r,r,16,1], [1,16,r,r,r,12,1]]
# ffn_rank = [[1,r,r,r,r,r,1], [1,r,r,r,r,r,1]]

ffn_tensor_type = 'TensorTrain'

# attention_shape = [[4,4,8,6,4,4,8,6],[4,4,8,6,4,4,8,6]]
# attention_rank = [r,r]
# attention_tensor_type = 'TensorTrain'

# ffn_shape = [[4,4,8,6,6,8,8,8],[6,8,8,8,4,4,8,6]]
# ffn_rank = [r,r]
# ffn_tensor_type = 'TensorTrain'

bit_attn = 8
scale_attn = 2**(-5)
bit_ffn = 8
scale_ffn = 2**(-5)
bit_a = 8
scale_a = 2**(-5)

# bit_attn = 32
# scale_attn = 2**(-0)
# bit_ffn = 32
# scale_ffn = 2**(-0)
# bit_a = 32
# scale_a = 2**(-0)

# bit_attn = 4
# scale_attn = 2**(-5)
# bit_ffn = 4
# scale_ffn = 2**(-5)
# bit_a = 4
# scale_a = 2**(-5)

# bit_attn = 2
# scale_attn = 2**(-5)
# bit_ffn = 2
# scale_ffn = 2**(-5)
# bit_a = 2
# scale_a = 2**(-5)


d_classifier=768
num_class = 22 #ATIS
slot_num = 121 #ATIS

# num_class = 60 #en
# slot_num = 56 #en
dropout_classifier = 0.1




# slot_num = 120
# num_class = 21


# classifier_shape = [4,4,8,6,4,4,8,6]
classifier_shape = [12,8,8,8,8,12]
classifier_rank = [1,12,r,r,r,12,1]
# classifier_rank = [1,r,r,r,r,r,1]

classifier_tensor_type = 'TensorTrain'

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


def main():

    PATH = 'model/test_ATIS_tensor_2layer_INT4.chkpt'
    PATH = 'model/test.chkpt'
    PATH = 'model/ATIS_tensor_2layers_FP8_input52_factor52_med52_grad52_dynamic_1e-3_test.chkpt'
    PATH = 'model/ATIS_tensor_2layers_FP8_all52_dynamic_1e-3_new.chkpt'
    PATH = 'model/test_ZO.chkpt'
    # PATH = 'models_end2end/MNLI_r20_nopre.chkpt'
    # transformer.load_state_dict(torch.load(PATH))

    epochs = 40

    # torch.save(transformer.encoder.src_word_emb,'model/embedding.chkpt')
    # torch.save(transformer.classifier,'model/classifier.chkpt')
    # torch.save(transformer.slot_classifier,'model/slot_classifier.chkpt')
    # torch.save(transformer.encoder.layer_stack,'model/encoder.chkpt')



    # valid_loss, valid_accu, valid_slot_accu = eval_epoch(transformer, validation_data, device, weight=None)
    block = transformer.encoder.layer_stack[0].slf_attn.w_qs




    precondition = (opt.precondition==1)
    
    
    step = 2

    # transformer.to(torch.bfloat16)
    par = list(transformer.parameters())
    # for p in par:
    #     p.requires_grad = True


    lr = 1e-2

    # for layer in transformer.modules():
    #     if hasattr(layer, 'tensor'):
    #         print(layer.parameters())

    # transformer.to(torch.float16)

    # optimizer = optim.Adam(par,betas=(0.9, 0.98), eps=1e-6, lr = lr)

    # optimizer = ScheduledOptim(
    #                 optim.Adam(
    #                     filter(lambda x: x.requires_grad, transformer.parameters()),
    #                     betas=(0.9, 0.98), eps=1e-09, lr = 1e-4),
    #                 opt.d_model, opt.n_warmup_steps)
    # optimizer_tensor = None

    # optimizer = optim.Adam(
    #                     filter(lambda x: x.requires_grad, transformer.parameters()),
    #                     betas=(0.9, 0.98), eps=1e-09, lr = 1e-3)
    # optimizer_tensor = None
    if precondition == True:
        optimizer_tensor = torch.optim.SGD(tensor_blocks.parameters(),lr = 1e-1)
        optimizer = optim.Adam(filter(lambda x: x.requires_grad, layer_notensor.parameters()),
                    betas=(0.9, 0.98), eps=1e-09, lr = 1e-3)
    else:
        if tensorized == True:
            # optimizer = ScheduledOptim(
            #             optim.Adam(
            #                 filter(lambda x: x.requires_grad, transformer.parameters()),
            #                 betas=(0.9, 0.98), eps=1e-09, lr = 1e-4),
            #             opt.d_model, opt.n_warmup_steps)
            # optimizer_tensor = optim.Adam(
            #                 tensor_blocks.parameters(),
            #                 betas=(0.9, 0.98), eps=1e-06, lr = 1e-3)
            
            # optimizer_tensor = torch.optim.SGD(transformer.parameters(),lr = 1e-1)

            lr = 1e-3
            # optimizer_tensor = torch.optim.SGD(tensor_blocks.parameters(),lr = 1e-1)
            optimizer_tensor = optim.Adam(filter(lambda x: x.requires_grad, tensor_blocks.parameters()),betas=(0.9, 0.98), eps=1e-06, lr = lr)
            optimizer = optim.Adam(layer_notensor.parameters(), betas=(0.9, 0.98), eps=1e-06, lr = lr)

            # optimizer = torch.optim.SGD(transformer.parameters(),lr = 1e-2)
            # optimizer_tensor = None

            # from ZO_optimizer import ZO_Optimizer
            # Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            # optimizer_ZO = ZO_Optimizer(transformer, Loss, model_old=transformer_old, n_sample=10,sigma=1e-1, device='cuda',sign=False)

            optimizer_ZO = None

            # optimizer = torch.optim.SGD(transformer.parameters(),lr = 1e-1)
            # optimizer_tensor = None
            # from ZO_SGD import ZO_SGD_Optimizer
            # Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
            # optimizer = ZO_SGD_Optimizer(transformer, Loss, model_old=transformer_old, lr = 1e-3, signSGD = False, n_sample=10, device='cuda')
            # optimizer_tensor = None


            # optimizer = optim.Adam(transformer.parameters(), betas=(0.9, 0.98), eps=1e-04, lr = 1e-3)

            # 

            # optimizer = optim.Adam(filter(lambda x: x.requires_grad, layer_notensor.parameters()),
                    # betas=(0.9, 0.98), eps=1e-06, lr = 0)
        else:
            optimizer = optim.Adam(
                            filter(lambda x: x.requires_grad, transformer.parameters()),
                            betas=(0.9, 0.98), eps=1e-06, lr = 1e-4)
            optimizer_tensor = None
    valid_acc_all = [-1]
    train_result = []
    test_result = []
    for epoch in range(epochs):
        start = time.time()



        train_loss, train_accu, train_slot_accu = train_epoch_bylayer(transformer, training_data, optimizer,optimizer_ZO=optimizer_ZO ,optimizer_tensor=optimizer_tensor,precondition=precondition,device=device,tensor_blocks=tensor_blocks,step=epoch)

        # train_loss, train_accu = 0,0

        start_val = time.time()

        valid_loss, valid_accu, valid_slot_accu = eval_epoch(transformer, validation_data, device, weight=None)

        test_loss, test_accu, test_slot_accu = eval_epoch(transformer, test_data, device, weight=None)
        # train_loss_new, train_accu = eval_epoch(transformer, training_data, device, weight=None)
        train_result += [train_loss.cpu().to(torch.float32),train_accu.cpu().to(torch.float32),train_slot_accu.cpu().to(torch.float32)]
        test_result += [test_loss,test_accu.cpu().to(torch.float32),test_slot_accu]


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
        
        full_model_name = opt.save_model + '.chkpt'
        torch.save(transformer.state_dict(),full_model_name)

        if max(valid_acc_all)<valid_accu:
            best_model_name = opt.save_model + '_best' + '.chkpt'
            torch.save(transformer.state_dict(),best_model_name)
        valid_acc_all.append(valid_accu)
    PATH_np = opt.save_model + '.npy'
    np.save(PATH_np,np.array([train_result,test_result]))




def train_epoch_bylayer(model, training_data, optimizer,optimizer_ZO=None,optimizer_tensor=None, tensor_blocks=None,precondition=False,device='cuda',step=1):
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

    Loss = nn.CrossEntropyLoss(label_smoothing=0.1)
    for batch in tqdm(
            training_data, mininterval=2,
            desc='  - (Training)   ', leave=False):

        target, w1, slot_label,attn,seg= map(lambda x: x.to(device), batch)


        optimizer.zero_grad()
        # print(optimizer.lr)
        if optimizer_tensor!=None:
            optimizer_tensor.zero_grad()

        # attn = None
        # model.eval()
        pred,pred_slot = model(w1,attn=attn,seg=seg)
        # pred_2,pred_slot_2 = model(w1,attn=attn,seg=seg)

        # print('diff=',torch.max(pred-pred_2))

        pred_slot = torch.flatten(pred_slot,start_dim=0, end_dim=1)

        slot_label = torch.flatten(slot_label,start_dim=0, end_dim=1)



        
  

        loss_MLM =  Loss(pred_slot, slot_label)
        loss = Loss(pred,target)  + loss_MLM



        memory = torch.cuda.max_memory_allocated()/1024/1024/1024
        max_memory = max(max_memory,memory)

        if optimizer_ZO!=None:
            optimizer_ZO.estimate_grad((w1,attn,seg),(target,slot_label))
            
        else:
            # print(model.encoder.layer_stack[0].slf_attn.w_ks.scale_med.grad)
            loss.backward()
            
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


        # update parameters
        # print(model.encoder.layer_stack[0].slf_attn.w_ks.tensor.factors[0])
        ####gradient based optimizer        
        optimizer.step()
        if optimizer_tensor!=None:
            optimizer_tensor.step()
        # print(model.encoder.layer_stack[0].slf_attn.w_ks.tensor.factors[0])
        # print(model.encoder.layer_stack[0].slf_attn.w_ks.tensor.factors[0].grad)
        
        # break
        # print(model.encoder.layer_stack[0].slf_attn.w_ks.scale_med.grad)
        # print(model.encoder.layer_stack[0].slf_attn.w_ks.tensor.factors[0].grad)


        #ZO order optimizer
        # print(loss)
        # optimizer.step((w1,attn,seg),(target,slot_label))
        
 

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
