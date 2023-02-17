import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import TensorizedLinear_module
import tensorly as tl




class EncoderLayer(nn.Module):
    ''' Compose with two layers '''

    def __init__(self, d_model, d_inner, n_head, d_q,d_k, d_v, dropout=0.1,
                attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
                ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
                bit_attn = 8, scale_attn = 2**(-5), 
                bit_ffn = 8, scale_ffn = 2**(-5),
                bit_a = 8, scale_a = 2**(-5),
                quantized = False,
                tensorized=True):
        super(EncoderLayer, self).__init__()
        
        if tensorized:
            self.slf_attn = Tensor_Attention(d_model,d_q,d_k,d_v,n_head, dropout=dropout,shape=attention_shape,rank=attention_rank,tensor_type=attention_tensor_type,
            bit_w=bit_attn,scale_w=scale_attn,
            quantized=quantized)
            self.pos_ffn = Tensor_PFF(d_model, d_inner, dropout=dropout,shape=ffn_shape,rank=ffn_rank,tensor_type=ffn_tensor_type,
            bit_w=bit_ffn,scale_w=scale_ffn,
            bit_a=bit_a, scale_a=scale_a,
            quantized=quantized)
        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)
        
        
       
    def forward(self, enc_input, slf_attn_mask=None, ranks=None,scale=None, use_kl = False,prune=False):
        enc_output, enc_slf_attn, kl_1 = self.slf_attn(
            enc_input, enc_input, enc_input, mask=slf_attn_mask, ranks=ranks, scale=scale, use_kl = use_kl,prune=prune)
        enc_output, kl_2 = self.pos_ffn(enc_output,ranks=ranks,scale=scale, use_kl = use_kl,prune=prune)


        return enc_output, enc_slf_attn, kl_1+kl_2


class DecoderLayer(nn.Module):
    ''' Compose with three layers '''

    def __init__(self, d_model, d_inner, n_head, d_q,d_k, d_v, dropout=0.1,
                attention_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], attention_rank = [20,20],attention_tensor_type = 'TensorTrain',
                ffn_shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], ffn_rank = [20,20],ffn_tensor_type = 'TensorTrain',
                bit_attn = 8, scale_attn = 2**(-5), 
                bit_ffn = 8, scale_ffn = 2**(-5),
                bit_a = 8, scale_a = 2**(-5),
                quantized = False,
                tensorized=True):
        super(DecoderLayer, self).__init__()

        
        if tensorized:
            self.slf_attn = Tensor_Attention(d_model,d_q,d_k,d_v,n_head, dropout=dropout,shape=attention_shape,rank=attention_rank,tensor_type=attention_tensor_type,
            bit_w=bit_attn,scale_w=scale_attn,
            quantized=quantized)

            self.enc_attn = Tensor_Attention(d_model,d_q,d_k,d_v,n_head, dropout=dropout,shape=attention_shape,rank=attention_rank,tensor_type=attention_tensor_type,
            bit_w=bit_attn,scale_w=scale_attn,
            quantized=quantized)

            self.pos_ffn = Tensor_PFF(d_model, d_inner, dropout=dropout,shape=ffn_shape,rank=ffn_rank,tensor_type=ffn_tensor_type,
            bit_w=bit_ffn,scale_w=scale_ffn,
            bit_a=bit_a, scale_a=scale_a,
            quantized=quantized)

            # self.slf_attn = Tensor_Attention(d_model,d_q,d_k,d_v,n_head, dropout=dropout,shape=attention_shape,rank=attention_rank,tensor_type=attention_tensor_type)
            # self.enc_attn = Tensor_Attention(d_model,d_q,d_k,d_v,n_head, dropout=dropout,shape=attention_shape,rank=attention_rank,tensor_type=attention_tensor_type)
            # self.pos_ffn = Tensor_PFF(d_model, d_inner, dropout=dropout,shape=ffn_shape,rank=ffn_rank,tensor_type=ffn_tensor_type)

        else:
            self.slf_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.enc_attn = MultiHeadAttention(n_head, d_model, d_k, d_v, dropout=dropout)
            self.pos_ffn = PositionwiseFeedForward(d_model, d_inner, dropout=dropout)

        

    def forward(self, dec_input, enc_output,
                slf_attn_mask=None, dec_enc_attn_mask=None):
        dec_output, dec_slf_attn = self.slf_attn(
            dec_input, dec_input, dec_input, mask=slf_attn_mask)
        dec_output, dec_enc_attn = self.enc_attn(
            dec_output, enc_output, enc_output, mask=dec_enc_attn_mask)
        dec_output = self.pos_ffn(dec_output)
        return dec_output, dec_slf_attn, dec_enc_attn



class Tensor_PFF(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], rank=[20,20], dropout=0.1,tensor_type = 'TensorTrain',
                bit_w = 8, scale_w = 2**(-5), 
                bit_a = 8, scale_a = 2**(-5), 
                quantized = False):
        super().__init__()
        
        em_stepsize = 1.0

       
        if quantized == False:
            self.fc_1 = TensorizedLinear_module(d_in, d_hid, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize, bias=True)
            self.fc_2 = TensorizedLinear_module(d_hid, d_in, shape=shape[1], tensor_type=tensor_type,max_rank=rank[1],em_stepsize=em_stepsize, bias=True)


            
            # self.relu = nn.ReLU()
            self.relu = nn.GELU()



        # self.relu = nn.GELU()


        
        self.layer_norm = nn.LayerNorm((d_in,), eps=1e-12)
        # self.layer_norm = nn.LayerNorm((d_in,), eps=1e-6)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x,ranks=None,scale=None,use_kl = False,prune=False):

        # print(torch.max(self.fc_1.tensor.factors[0]))
        kl_loss = 0
        if use_kl == True:
            kl_loss += self.fc_1.get_kl()
            kl_loss += self.fc_2.get_kl()


        residual = x

        x = self.fc_1(x,ranks,scale,prune=prune)
        x = self.relu(x)
        x = self.fc_2(x,ranks,scale,prune=prune)

        x = self.dropout(x)


        x = x + residual

        x = self.layer_norm(x)
        
        return x,kl_loss



class Tensor_Attention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, d_model,d_q,d_k,d_v, n_head, shape = [[4,4,8,4,4,4,8,4],[4,4,8,4,4,4,8,4]], rank=[20,20],tensor_type = 'TensorTrain', dropout=0.1,
                bit_w = 8, scale_w = 2**(-5), 
                quantized = False):
        super().__init__()


        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

      
        em_stepsize = 1.0

        if quantized == False:
            self.w_qs = TensorizedLinear_module(d_model, d_q*n_head, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize,bias=True)
            self.w_ks = TensorizedLinear_module(d_model, d_k*n_head, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize,bias=True)
            self.w_vs = TensorizedLinear_module(d_model, d_v*n_head, shape=shape[0], tensor_type=tensor_type,max_rank=rank[0],em_stepsize=em_stepsize,bias=True)

            self.fc = TensorizedLinear_module(d_v*n_head, d_model, shape=shape[1], tensor_type=tensor_type,max_rank=rank[1],em_stepsize=em_stepsize,bias=True)



        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)


        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm((d_model,), eps=1e-12)
        # self.layer_norm = nn.LayerNorm((d_model,), eps=1e-6)


    

    def forward(self, q, k, v, mask=None, ranks=None,scale=None,use_kl=False,prune=False):

        kl_loss = 0
        if use_kl == True:
            kl_loss += self.w_qs.get_kl()
            kl_loss += self.w_ks.get_kl()
            kl_loss += self.w_vs.get_kl()
            kl_loss += self.fc.get_kl()




        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

 

        q = self.w_qs(q,ranks,scale,prune=prune).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k,ranks,scale,prune=prune).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v,ranks,scale,prune=prune).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
        




        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)




        q = self.dropout(self.fc(q,ranks,scale,prune=prune))


        
        q += residual



        q = self.layer_norm(q)


        return q, attn, kl_loss


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))


        # print(mask.shape)
        if mask!=None and len(mask.shape)==3:
            mask.to(torch.float32)
            mask = mask.unsqueeze(1)
            mask = (1-mask)*(-1e9)
            # mask = torch.unsqueeze(mask,-1).to(torch.float)
            # mask = mask@ torch.transpose(mask,-1,-2)
        
        

        
        if mask is not None:
            attn = attn.to(torch.float32) + mask
            # attn = attn.masked_fill(mask <= 0.1, -1e9)
            # attn = attn.masked_fill(mask <= 0.1, 0)
            # attn = torch.matmul(attn,mask)

        # print(mask.shape)

        attn_prob = F.softmax(attn, dim=-1).to(v.dtype)
        # print(attn_prob[0,0,:10])

        attn_prob = self.dropout(attn_prob)



        output = torch.matmul(attn_prob, v)
        # output = torch.matmul(attn, v)


        # print(output)

        return output, attn_prob


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v


        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=True)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=True)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=True)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None, ranks=None,scale=None, use_kl = False,prune=False):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv


        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.
     
        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn,0


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        self.w_1 = nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

        self.relu = nn.GELU()

    def forward(self, x, ranks=None,scale=None, use_kl = False,prune=False):

        residual = x

        x = self.w_2(self.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x,0