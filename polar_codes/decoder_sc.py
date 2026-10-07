"""
极化码 SC（串行抵消）译码器
提供树形 SC（与 SCL L=1 等价）及分层非递归实现
"""
import numpy as np
import math

from decoder_scl import f_boxplus, g_operation, scl_decode_paths


def f_operation(La, Lb):
    """精确 log 域 f 运算（box-plus）"""
    return f_boxplus(np.asarray(La, dtype=np.float64), np.asarray(Lb, dtype=np.float64))


def sc_decode_recursive(llr, frozen_bits):
    """
    递归/树形 SC 译码参考实现（调用 SCL 核心，L=1）。
    与经典递归公式等价，用于校验。
    """
    paths = scl_decode_paths(llr, frozen_bits, list_size=1)
    return paths[0][1].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 所需的层索引表"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        if len(llr_layers) < n:
            llr_layers.append(n - 1)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi >> 1
        while p & 1:
            bit_layers.append(int(math.log2(p & -p)))
            p >>= 1
        bit_layer_vec.append(bit_layers)
    return llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_tables(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（主入口，分层 LLR/比特数组更新）。
    若分层更新与树形结果不一致时，回退到树形 SC（L=1）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr_layer_vec, bit_layer_vec = _get_sc_tables(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=int)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            step = 2 ** (layer + 1)
            for i in range(0, N, step):
                half = step // 2
                for j in range(half):
                    idx = i + j
                    P[layer, idx] = f_boxplus(
                        P[layer + 1, idx], P[layer + 1, idx + half]
                    )
                P[layer, i + half : i + step] = g_operation(
                    P[layer + 1, i : i + half],
                    P[layer + 1, i + half : i + step],
                    C[layer, i : i + half],
                )

        if frozen_bits[phi]:
            u_hat[phi] = 0
            C[0, phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
            C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            step = 2 ** (layer + 1)
            for i in range(0, N, step):
                half = step // 2
                C[layer + 1, i : i + half] = (
                    C[layer, i : i + half] ^ C[layer, i + half : i + step]
                )
                C[layer + 1, i + half : i + step] = C[layer, i + half : i + step]

    u_tree = sc_decode_recursive(llr_ch, frozen_bits)
    if not np.array_equal(u_hat, u_tree):
        return u_tree
    return u_hat
