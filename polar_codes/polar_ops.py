"""极化码 SC/SCL 共用的 f/g 运算与索引预计算"""
import math
import numpy as np


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_f = []
        t = phi + 1
        layer = 0
        while t % 2 == 0 and layer < n:
            layers_f.append(layer)
            t //= 2
            layer += 1
        llr_layer_vec.append(layers_f)

        layers_b = []
        t = phi
        layer = 0
        while t % 2 == 1 and layer < n:
            layers_b.append(layer)
            t //= 2
            layer += 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec
