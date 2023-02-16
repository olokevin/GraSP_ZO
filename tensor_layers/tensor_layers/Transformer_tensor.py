import torch
import torch.nn as nn
import numpy as np

from .Transformer_tensor_sublayers import EncoderLayer, DecoderLayer
from .layers import TensorizedEmbedding, TensorizedLinear_module



def get_pad_mask(seq, pad_idx):
    return (seq != pad_idx).unsqueeze(-2)


def get_subsequent_mask(seq):
    ''' For masking out the subsequent info. '''
    sz_b, len_s = seq.size()
    subsequent_mask = (1 - torch.triu(
        torch.ones((1, len_s, len_s), device=seq.device), diagonal=1)).bool()
    return subsequent_mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_hid, n_position=200):
        super(PositionalEncoding, self).__init__()

        # Not a parameter
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_position, d_hid))

    def _get_sinusoid_encoding_table(self, n_position, d_hid):
        ''' Sinusoid position encoding table '''
        # TODO: make it with torch instead of numpy

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        # print(self.pos_table[:, :x.size(1),:2])
        return x + self.pos_table[:, :x.size(1)].clone().detach()

class Encoder(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=True,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True,
            uncompressed=0,
            uncompressed_last = 0,
            replace = False):

        super().__init__()

        self.n_layers = n_layers
        

        if tensorized:
            self.src_word_emb = TensorizedEmbedding(
                    tensor_type=emb_tensor_type,
                    max_rank=emb_rank,
                    shape=emb_shape,
                    prior_type='log_uniform',
                    eta=None)
        else:
            self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)

        # self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec)


        # self.src_word_emb = model_origin._modules['distilbert']._modules['embeddings']
      
        # self.position_enc = PositionalEncoding(d_word_vec, n_position=n_position)
        self.position_enc = nn.Embedding(n_position,d_word_vec)
        self.token_type_emb = nn.Embedding(2,d_word_vec)

        if replace == True:
            self.src_word_emb_BERT = nn.Embedding(n_src_vocab, d_word_vec)
            self.position_enc_BERT = nn.Embedding(n_position,d_word_vec)
            self.token_type_emb_BERT = nn.Embedding(2,d_word_vec)
            self.layer_norm_BERT = nn.LayerNorm(d_model, eps=1e-12)

        self.register_buffer("position_ids", torch.arange(n_position).expand((1, -1)))

        self.dropout = nn.Dropout(p=dropout)
        if len(attention_rank)==2:
            attention_rank = [attention_rank]*n_layers
        if len(ffn_rank)==2:
            ffn_rank = [ffn_rank]*n_layers


        # if uncompressed==0:
        #     self.layer_stack = nn.ModuleList([
        #         EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
        #         attention_shape=attention_shape,attention_rank=attention_rank[i],attention_tensor_type=attention_tensor_type,
        #         ffn_shape=ffn_shape,ffn_rank=ffn_rank[i],ffn_tensor_type=ffn_tensor_type,
        #         tensorized=tensorized,
        #         bit_attn = bit_attn, scale_attn = scale_attn, 
        #         bit_ffn = bit_ffn, scale_ffn = scale_ffn,
        #         bit_a = bit_a, scale_a = scale_a,
        #         quantized=quantized)
        #         for i in range(n_layers)])
        # else:
        if replace == True:
            self.layer_stack_BERT = nn.ModuleList([
                EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
                attention_shape=attention_shape,attention_rank=attention_rank[i],attention_tensor_type=attention_tensor_type,
                ffn_shape=ffn_shape,ffn_rank=ffn_rank[i],ffn_tensor_type=ffn_tensor_type,
                tensorized=False,
                bit_attn = bit_attn, scale_attn = scale_attn, 
                bit_ffn = bit_ffn, scale_ffn = scale_ffn,
                bit_a = bit_a, scale_a = scale_a,
                quantized=quantized)
                for i in range(n_layers)])
        else:
            self.layer_stack_BERT = None


        self.layer_stack = [
            EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
            attention_shape=attention_shape,attention_rank=attention_rank,attention_tensor_type=attention_tensor_type,
            ffn_shape=ffn_shape,ffn_rank=ffn_rank,ffn_tensor_type=ffn_tensor_type,
            tensorized=False,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized)
            for _ in range(uncompressed)]
        self.layer_stack += [EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
            attention_shape=attention_shape,attention_rank=attention_rank[i],attention_tensor_type=attention_tensor_type,
            ffn_shape=ffn_shape,ffn_rank=ffn_rank[i],ffn_tensor_type=ffn_tensor_type,
            tensorized=tensorized,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized)
            for i in range(uncompressed,n_layers-uncompressed_last)]
        self.layer_stack += [EncoderLayer(d_model, d_inner, n_head,d_q, d_k, d_v, dropout=dropout,
            attention_shape=attention_shape,attention_rank=attention_rank[i],attention_tensor_type=attention_tensor_type,
            ffn_shape=ffn_shape,ffn_rank=ffn_rank[i],ffn_tensor_type=ffn_tensor_type,
            tensorized=False,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized)
            for i in range(n_layers-uncompressed_last,n_layers)]
        self.layer_stack = nn.ModuleList(self.layer_stack)

        self.layer_stack_list = [self.layer_stack,self.layer_stack_BERT]

        self.layer_norm = nn.LayerNorm(d_model, eps=1e-12)
        self.scale_emb = scale_emb
        self.d_model = d_model

    def forward(self, src_seq, attn=None, seg=None, ranks=None, scale=None, layer=11, use_kl=False,prune=False,replace=None):
        if replace==None:
            replace = [0]*(self.n_layers+1)

        position_ids = self.position_ids[:, 0: src_seq.shape[1]]
        if seg==None:
                seg = torch.zeros(attn.shape).to(int).to(attn.device)

        enc_slf_attn_list = []

        # -- Forward
        if replace[-1]==0:
            enc_output = self.src_word_emb(src_seq)
            enc_output = enc_output+self.position_enc(position_ids)  +self.token_type_emb(seg) # forget to add token_type_emb before 1/29/2023
            enc_output = self.layer_norm(enc_output)
        else:
            enc_output = self.src_word_emb_BERT(src_seq)
            enc_output = enc_output+self.position_enc_BERT(position_ids)  +self.token_type_emb_BERT(seg)
            enc_output = self.layer_norm_BERT(enc_output)

        
        # if self.scale_emb:
        #     enc_output *= self.d_model ** 0.5
        

        

        
        enc_output = self.dropout(enc_output)

        kl_loss = 0
        
        for i in range(min(layer+1,len(self.layer_stack))):
            enc_layer = self.layer_stack_list[replace[i]][i]
            if ranks!=None:
 
                enc_output, enc_slf_attn,kl = enc_layer(enc_output, slf_attn_mask=attn, ranks=ranks[i],scale=scale,use_kl=use_kl,prune=prune)
            else:
                enc_output, enc_slf_attn,kl = enc_layer(enc_output, slf_attn_mask=attn,use_kl=use_kl,prune=prune)
            kl_loss += kl

            # enc_slf_attn_list += [enc_slf_attn] if return_attns else []
        return enc_output,kl_loss


class Transformer_sentence_concat_SLU(nn.Module):
    ''' A encoder model with self attention mechanism. '''

    def __init__(
            self, n_src_vocab,d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=0.1, n_position=200, scale_emb=False,
            emb_shape = [[16,16,8,5],[4,4,8,4]],emb_rank = 16, emb_tensor_type = 'TensorTrainMatrix',
            attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
            ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
            d_classifier=512,num_class = 3, dropout_classifier = 0.2,
            classifier_shape = [4,4,8,4,4,4,8,4],classifier_rank = 20,classifier_tensor_type = 'TensorTrain',
            bit_attn = 8, scale_attn = 2**(-5), 
            bit_ffn = 8, scale_ffn = 2**(-5),
            bit_a = 8, scale_a = 2**(-5),
            quantized = False,
            tensorized=True,
            slot_num = 2,
            uncompressed=0,
            uncompressed_last = 0):

        super().__init__()
        
        self.encoder = Encoder(n_src_vocab, d_word_vec, n_layers, n_head, d_q, d_k, d_v,
            d_model, d_inner, pad_idx, dropout=dropout, n_position=n_position, scale_emb=scale_emb,
            emb_shape = emb_shape,emb_rank = emb_rank, emb_tensor_type = emb_tensor_type,
            attention_shape = attention_shape, attention_rank = attention_rank,attention_tensor_type = attention_tensor_type,
            ffn_shape = ffn_shape, ffn_rank = ffn_rank,ffn_tensor_type = ffn_tensor_type,
            bit_attn = bit_attn, scale_attn = scale_attn, 
            bit_ffn = bit_ffn, scale_ffn = scale_ffn,
            bit_a = bit_a, scale_a = scale_a,
            quantized=quantized,
            tensorized=tensorized,
            uncompressed=uncompressed,
            uncompressed_last = uncompressed_last)

        if tensorized == True:
            self.preclassifier = TensorizedLinear_module(d_model, d_model, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)
            self.slot_preclassifier = TensorizedLinear_module(d_model, d_model, shape=classifier_shape, tensor_type=classifier_tensor_type,max_rank=classifier_rank)

        else:
            self.preclassifier = nn.Linear(d_model,d_model)
            self.slot_preclassifier = nn.Linear(d_model,d_model)

        
        # self.preclassifier = nn.Linear(d_model,d_model)

        self.classifier = nn.Sequential(self.preclassifier,nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))

        self.slot_classifier = nn.Sequential(self.slot_preclassifier,nn.GELU(),nn.LayerNorm(d_model),nn.Linear(d_model,slot_num))

        # self.classifier = nn.Sequential(nn.Linear(d_model,d_model),nn.Tanh(),nn.Dropout(p=dropout_classifier),nn.Linear(d_model,num_class))


        

        

  

    def forward(self, w1, attn=None,seg=None,ranks=None,scale=None,use_kl=False):
        # device = self.classifier[0].weight.device

        output,kl_loss = self.encoder(w1,attn,seg=seg,ranks=ranks,scale=scale,use_kl=use_kl)


        # output = torch.squeeze(output[:,0,:])


        
        # output = self.classifier(output)
        
        return self.classifier(output[:,0,:]), self.slot_classifier(output[:,1:,:])
        # return self.classifier(torch.mean(output,dim=1)), self.slot_classifier(output[:,1:,:])

        return output,kl_loss

