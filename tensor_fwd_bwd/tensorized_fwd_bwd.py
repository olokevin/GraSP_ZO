import torch
import numpy as np
import torch.nn.functional as F
from torch import tensordot, bmm
import tensorly as tl


# Author: Alvin Liu

def tt_times_matrix_fwd(tensor, matrix):
    ndims = tensor.order
    d = int(ndims / 2)
    ranks = tensor.rank
    tt_shape = tensor.shape
    tt_shape_row = list(tt_shape[:d])
    tt_shape_col = list(tt_shape[d:])
    matrix_cols = matrix.shape[0]

    saved_tensors = [matrix]
    left = []
    right = []

    output = tensor.factors[0].reshape(-1, ranks[1])
    left.append(output)
    for core in tensor.factors[1:d]:
        output = tensordot(output, core, dims=([-1], [0]))
        left.append(output)

    output = F.linear(matrix, torch.movedim(output.reshape(np.prod(tt_shape_row), -1), -1, 0))
    saved_tensors.append(left)

    temp = tensor.factors[d]
    right.append(temp)
    for core in tensor.factors[d + 1:]:
        temp = tensordot(temp, core, dims=([-1], [0]))
        right.append(temp)

    output = F.linear(output, torch.movedim(temp.reshape(ranks[d], np.prod(tt_shape_col)),
                                            0, -1)).reshape(matrix_cols, np.prod(tt_shape_col))
    saved_tensors.append(right)

    return output


def tt_times_matrix_bwd(tensor, dy, saved_tensors):
    ndims = tensor.order
    d = int(ndims / 2)
    ranks = tensor.rank
    tt_shape = tensor.shape
    tt_shape_row = list(tt_shape[:d])
    tt_shape_col = list(tt_shape[d:])
    matrix = saved_tensors[0]
    left = saved_tensors[1]
    right = saved_tensors[2]
    left_grads = []
    right_grads = []

    with torch.no_grad():
        dy_core_prod = right[-1]
        dy_core_prod = tensordot(dy, dy_core_prod.reshape(dy_core_prod.shape[0], -1), dims=([1], [1]))
        matrix_dy_core_prod = tensordot(matrix, dy_core_prod, dims=([0], [0]))

        for i in reversed(range(1, d)):
            grad = tensordot(left[i - 1].reshape(-1, ranks[i]),
                             matrix_dy_core_prod.reshape(np.prod(tt_shape_row[:i]), tt_shape_row[i], -1,
                                                         ranks[d]),
                             dims=([0], [0]))
            if i == d - 1:
                right_core = tensor.factors[i]
            else:
                grad = tensordot(grad, right_core, dims=([2, 3], [1, 2]))
                right_core = tensordot(tensor.factors[i], right_core,
                                       dims=([-1], [0])).reshape(ranks[i], -1, ranks[d])
            if grad.shape != tensor.factors[i].shape:
                grad = grad.reshape(list(tensor.factors[i].shape))

            left_grads.append(grad)

        left_grads.append(tensordot(matrix_dy_core_prod.reshape(tt_shape_row[0], -1, ranks[d]),
                                    right_core, dims=([1, 2], [1, 2])).reshape(1, tt_shape_row[0], -1))

        left_grads = left_grads[::-1]

        matrix_core_prod = left[-1]
        matrix_core_prod = tensordot(matrix_core_prod.reshape(-1, matrix_core_prod.shape[-1]),
                                     matrix, dims=([0], [1]))
        matrix_dy_core_prod = tensordot(matrix_core_prod, dy, dims=([1], [0]))

        for i in reversed(range(1, d)):
            grad = tensordot(right[i - 1].reshape(-1, ranks[d + i]),
                             matrix_dy_core_prod.reshape(-1, tt_shape_col[i], int(np.prod(tt_shape_col[i + 1:]))),
                             dims=([0], [0]))

            if i == d - 1:
                right_core = tensor.factors[d + i].reshape(-1, tt_shape_col[i])
            else:
                grad = tensordot(grad, right_core, dims=([-1], [1]))
                right_core = tensordot(tensor.factors[d + i], right_core, dims=([-1], [0])).reshape(ranks[d + i],
                                                                                                    -1)
            if grad.shape != tensor.factors[d + i].shape:
                grad = grad.reshape(list(tensor.factors[i].shape))

            right_grads.append(grad)

        right_grads.append(tensordot(matrix_dy_core_prod.reshape(ranks[d], tt_shape_col[0], -1),
                                     right_core, dims=([-1], [1])))

        right_grads = right_grads[::-1]

        dx = tensor.factors[-1].reshape(ranks[-2], -1)
        for core in reversed(tensor.factors[d:-1]):
            dx = tensordot(core, dx, dims=([-1], [0]))

        dx = tensordot(dy, dx.reshape(-1, np.prod(tt_shape_col)), dims=([-1], [-1]))

        temp = tensor.factors[0].reshape(-1, ranks[1])
        for core in tensor.factors[1:d]:
            temp = tensordot(temp, core, dims=([-1], [0]))

        dx = tensordot(dx, temp.reshape(np.prod(tt_shape_row), -1), dims=([-1], [-1]))

    return left_grads + right_grads, dx


def cp_fusion_fwd(vec_list, weight_tensor, outer_order):
    if not outer_order:
        outer_order = len(vec_list)

    if outer_order == 1:
        raise ValueError(f'At least order 2 outer product is expected, got 1.')

    output = tensordot(vec_list[0], weight_tensor.factors[0], dims=([-1], [0]))

    for i in range(1, outer_order):
        if len(vec_list) != outer_order:
            output *= tensordot(vec_list[0], weight_tensor.factors[i], dims=([-1], [0]))
        else:
            output *= tensordot(vec_list[i], weight_tensor.factors[i], dims=([-1], [0]))

    d = weight_tensor.order
    # kr_prod = tl.tenalg.khatri_rao([weight_tensor.factors[i] for i in range(outer_order, d)]).unsqueeze(0)
    kr_prod = tl.tenalg.khatri_rao([weight_tensor.factors[i] for i in range(outer_order, d)])
    # output = torch.sum(output.unsqueeze(1) * kr_prod, -1)

    output = tensordot(output, kr_prod, dims=([-1], [-1]))

    return output


def tt_fusion_fwd(vec_list, weight_tensor, outer_order):
    if not outer_order:
        outer_order = len(vec_list)

    if outer_order == 1:
        raise ValueError(f'At least order 2 outer product is expected, got 1.')

    tt_rank = weight_tensor.rank
    d = weight_tensor.order

    left = tensordot(vec_list[0], weight_tensor.factors[0].squeeze(), dims=([-1], [0])).unsqueeze(1)

    for i in range(1, outer_order):
        if len(vec_list) != outer_order:
            left = bmm(left, tensordot(vec_list[0], weight_tensor.factors[i], dims=([-1], [1])))
        else:
            left = bmm(left, tensordot(vec_list[i], weight_tensor.factors[i], dims=([-1], [1])))

    right = tensordot(weight_tensor.factors[outer_order], weight_tensor.factors[outer_order + 1], dims=([-1], [0]))
    for i in range(outer_order + 1, d - 1):
        right = tensordot(right, weight_tensor.factors[i + 1], dims=([-1], [0]))

    output = left.squeeze() @ right.reshape(tt_rank[outer_order], -1)

    return output


def blocktt_fusion_fwd(vec_list, weight_tensor, outer_order):
    if not outer_order:
        outer_order = len(vec_list)

    if outer_order == 1:
        raise ValueError(f'At least order 2 outer product is expected, got 1.')

    ttm_rank = weight_tensor.rank
    d = weight_tensor.order
    n = vec_list[0].shape[0]

    output = tensordot(vec_list[0], weight_tensor.factors[0].squeeze(), dims=([-1], [0]))

    for i in range(1, outer_order):
        if len(vec_list) != outer_order:
            vec = vec_list[0]
        else:
            vec = vec_list[i]

        output = bmm(
            output,
            tensordot(vec, weight_tensor.factors[i], dims=([-1], [1])).reshape(n, ttm_rank[i], -1)
        ).reshape(n, -1, ttm_rank[i + 1])

    return output.squeeze()


def tensorized_fusion_fwd_init(factorization):
    factorization_list = ['tt', 'cp', 'blocktt']
    if factorization not in factorization_list:
        raise TypeError(f'Supported tensor formats are {factorization_list}, got {factorization} instead')

    fwd_dict = dict(cp=cp_fusion_fwd, tt=tt_fusion_fwd, blocktt=blocktt_fusion_fwd)
    return fwd_dict[factorization]


def tensorized_linear_fwd_init(factorization):
    factorization_list = ['tt']
    if factorization not in factorization_list:
        raise TypeError(f'Supported tensor formats are {factorization_list}, got {factorization} instead')

    fwd_dict = dict(tt=tt_times_matrix_fwd)
    return fwd_dict[factorization]


def tensorized_linear_bwd_init(factorization):
    factorization_list = ['tt']
    if factorization not in factorization_list:
        raise TypeError(f'Supported tensor formats are {factorization_list}, got {factorization} instead')

    fwd_dict = dict(tt=tt_times_matrix_bwd)
    return fwd_dict[factorization]
