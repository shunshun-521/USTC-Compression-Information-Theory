"""
极化码编解码核心（NumPy 实现，与 Sionna PolarEncoder/SCDecoder 约定一致）
"""
import math

import numpy as np


def bit_reversal_permutation(N):
    n = int(math.log2(N))
    return np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)])


def _gen_encode_indices(N):
    nb_stages = int(math.log2(N))
    ind_gather = np.ones((nb_stages, N + 1), dtype=np.int32) * N
    for s in range(nb_stages):
        ind_range = np.arange(N // 2)
        ind_dest = ind_range * 2 - np.mod(ind_range, 2**s)
        ind_origin = ind_dest + 2**s
        ind_gather[s, ind_dest] = ind_origin
    return ind_gather


def polar_encode(u):
    """
    极化码编码：Sionna 风格蝶形 XOR（G_N 生成矩阵一致）。
    输入 u 为长度 N 的完整源向量（信息位 + 冻结位）。
    """
    u = np.asarray(u, dtype=np.uint8)
    N = len(u)
    n = int(math.log2(N))
    x = np.zeros(N + 1, dtype=np.uint8)
    x[:N] = u
    ind_gather = _gen_encode_indices(N)
    for s in range(n):
        helper = ind_gather[s]
        x = np.bitwise_xor(x, x[helper])
    return x[:N].astype(int)


def build_generator_matrix(N):
    """通过编码单位向量构建 G_N"""
    G = np.zeros((N, N), dtype=int)
    for i in range(N):
        e = np.zeros(N, dtype=int)
        e[i] = 1
        G[i] = polar_encode(e)
    return G


def _cn_op(x, y):
    x_in = np.clip(x, -30.0, 30.0)
    y_in = np.clip(y, -30.0, 30.0)
    return np.log1p(np.exp(x_in + y_in)) - np.logaddexp(x_in, y_in)


def f_operation(La, Lb):
    """min-sum 近似 f 运算（对外接口保留）"""
    La = np.clip(La, -30.0, 30.0)
    Lb = np.clip(Lb, -30.0, 30.0)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _polar_decode_sc(llr_ch, frozen_ind):
    """Sionna 风格递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        fr1 = frozen_ind[:half]
        fr2 = frozen_ind[half:]

        x_llr1 = _cn_op(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc(x_llr1, fr1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc(x_llr2, fr2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_int = np.bitwise_xor(u_hat1_up.astype(np.int8), u_hat2_up.astype(np.int8))
        u_hat_up = np.concatenate([u_hat1_up_int.astype(np.float64), u_hat2_up])
        return u_hat, u_hat_up

    is_frozen = frozen_ind[0] == 1
    if is_frozen:
        u_hat = np.array([0.0])
    else:
        decision = 0.5 * (1.0 - np.sign(llr_ch[0]))
        u_hat = np.array([1.0 if decision == 0.5 else decision])
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits, use_minsum=True):
    """递归 SC 译码"""
    frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
    u_hat, _ = _polar_decode_sc(np.asarray(llr, dtype=np.float64), frozen_ind)
    return u_hat.astype(int)


def precompute_sc_indices(N):
    """预计算 SCL 译码辅助索引"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        llr_layers, bit_layers = [], []
        for i in range(n):
            if (phi >> i) & 1 == 0:
                llr_layers.append(i)
            else:
                break
        for i in range(n):
            if (phi >> i) & 1 == 1:
                bit_layers.append(i)
            else:
                break
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码接口"""
    return sc_decode_recursive(llr_ch, frozen_bits)
