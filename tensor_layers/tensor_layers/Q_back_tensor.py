from ctypes import Union
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from .low_rank_tensors import CP,TensorTrain,TensorTrainMatrix,Tucker
from . import low_rank_tensors
from .emb_utils import get_cum_prod,tensorized_lookup
import tensorly as tl

from math import prod


###################Define Quantized Backward############################################
class ttm_times_mat(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, quant,matrix, *factors):
        """
        In the forward pass we receive a Tensor containing the input and return
        a Tensor containing the output. ctx is a context object that can be used
        to stash information for backward computation. You can cache arbitrary
        objects for use in the backward pass using the ctx.save_for_backward method.
        """
        #compute matrix.T @ tensor(factors)
        # Prepare tensor shape
        shape_x = matrix.T.shape
        tensorized_shape_x = tuple([U.shape[1] for U in factors])
        tensorized_shape_y = tuple([U.shape[2] for U in factors])

        rank = [U.shape[0] for U in factors[0:]] + [1] 
        # tensorized_shape_x, tensorized_shape_y = tensor.tensorized_shape

        num_batch = shape_x[0]
        order = len(factors)

        # Reshape transpose of input matrix to input tensor

        input_tensor = torch.reshape(matrix.T, (shape_x[0], ) + tensorized_shape_x)

        # Compute left partial sum
        # saved_tensor[k+1] = x.T * G1 * ... * Gk
        # saved_tensor[k+1].shape: num_batch * (i1*...*ik-1) * ik * (jk+1*...*jd) * rk
        saved_tensors = []
        current_i_dim = 1
        saved_tensors.append(
            input_tensor.reshape(num_batch, current_i_dim, 1, -1, rank[0]))

        for k in range(order):
            current_tensor = saved_tensors[k]
            saved_tensors.append(
                torch.einsum(
                    'aibcd,dbef->aiecf',
                    current_tensor.reshape(num_batch, current_i_dim,
                                        tensorized_shape_x[k], -1,
                                        rank[k]), factors[k]))
            current_i_dim *= tensorized_shape_y[k]

        # Forward Pass
        # y[i1,...,id] = sum_j1_..._jd G1[i1,j1] * G2[i2,j2] * ... * Gd[id,jd] * x[j1,...,jd]
        output = saved_tensors[order].reshape(num_batch, -1)


        ctx.save = saved_tensors
        ctx.factors = factors
        ctx.quant = quant
        return output


    @staticmethod
    def backward(ctx, dy):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        Q = ctx.quant

        saved_tensors = ctx.save
        factors = ctx.factors

        tensorized_shape_x = tuple([U.shape[1] for U in factors])
        tensorized_shape_y = tuple([U.shape[2] for U in factors])

        rank = [U.shape[0] for U in factors] + [1]

        grads = [
        torch.zeros(x.shape, dtype=x.dtype, device=x.device)
        for x in factors]
        dx = torch.zeros(saved_tensors[0].shape,
                        dtype=saved_tensors[0].dtype,
                        device=saved_tensors[0].device)

        # Prepare tensor shape
        input_tensor = saved_tensors[0]
        num_batch = input_tensor.shape[0]
        order = len(factors)

        j_right = 1
        i_left = prod(tensorized_shape_y)

        for k in range(order - 1, -1, -1):
            j_right *= tensorized_shape_x[k]
            i_left //= tensorized_shape_y[k]

            left = saved_tensors[k]
            cur_j_right = j_right
            cur_i_right = 1

            for cur_k in range(order - 1, k, -1):
                cur_j = tensorized_shape_x[cur_k]
                cur_i = tensorized_shape_y[cur_k]
                cur_j_right //= cur_j
                left = torch.einsum(
                    'abicdeh,fdgh->abgicef',
                    left.reshape(num_batch, i_left, cur_i_right, cur_j_right,
                                cur_j, rank[k], rank[cur_k + 1]),factors[cur_k])
                cur_i_right *= cur_i

            # Contract with dy
            grads[k] = torch.einsum(
                generate_contraction_string(k, order),
                left.reshape(
                    (num_batch, ) + tensorized_shape_y[:k] +
                    tensorized_shape_y[k + 1:] + (
                        cur_j_right,
                        rank[k],
                        rank[k + 1],
                    )
                ),  # Shape: num_batch * i1 * ... * ik-1 * ik+1 * ... id * jk * rk * rk+1
                dy.reshape((num_batch, ) + tensorized_shape_y))

        # Compute dx
        saved_dx_tensors = []
        i_right = prod(tensorized_shape_y)
        j_left = 1
        saved_dx_tensors.append(dy.reshape((num_batch, ) + tensorized_shape_y))

        for k in range(order):
            i_right //= tensorized_shape_y[k]
            saved_dx_tensors.append(
                torch.einsum(
                    'xcijr,rdce->xijde',
                    saved_dx_tensors[k].reshape(num_batch, tensorized_shape_y[k],
                                                i_right, j_left, rank[k]),
                    factors[k]).reshape(num_batch, i_right, -1, rank[k + 1]))
            j_left *= tensorized_shape_x[k]
        dx = saved_dx_tensors[-1].squeeze().permute(1, 0)


        return None, Q(dx), *[Q(grad) for grad in grads]


        input, scale= ctx.saved_tensors
        grad_input = grad_output*torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), 1.0, 0.0)
        
        grad_scale = (torch.where((ctx.input_div_scale<=ctx.max_q) & (ctx.input_div_scale>=ctx.min_q), ctx.q_input - ctx.input_div_scale, ctx.input_div_scale))

        grad_scale = grad_output*torch.clamp(grad_scale, min = ctx.min_q, max = ctx.max_q)

        return grad_input, grad_scale, None, None, None



def generate_contraction_string(k, order):
    left = k
    right = order - k - 1
    CHAR = 'abcdefghijklmnopqrstuvwxyz'
    string = 'X' + CHAR[:left] + CHAR[
        left + 1:right + left +
        1] + 'JRS,X' + CHAR[:order] + '->RJ' + CHAR[k] + 'S'
    return string