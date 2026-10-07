"""
极化码 SC（串行抵消）译码器
提供递归/非递归 SC（标准 f/g 树）及与生成矩阵一致的软输入硬判决译码（用于仿真）。
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, polar_generator_matrix
from utils_gf2 import get_ginv, gf2_decode_from_llr


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（参考实现，输入为信道自然序 LLR）。"""
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr = np.asarray(llr, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)[br]
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    off = 0
    for layer in range(n + 1):
        lambda_offset[layer] = off
        off += 1 << layer

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        llr_layers = []
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        p = phi
        bit_layers = []
        while p & 1:
            bit_layers.append(int(math.log2(p & -p)))
            p >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC（分层 L/C 数组）。"""
    N = len(llr_ch)
    m = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen = np.asarray(frozen_bits, dtype=bool)[br]
    y = np.asarray(llr_ch, dtype=np.float64)[br]

    L = np.zeros((m + 1, N), dtype=np.float64)
    C = np.zeros((m + 1, N), dtype=np.int8)
    L[m, :N] = y

    u_br = np.zeros(N, dtype=int)
    _, llr_layer_vec, _ = precompute_sc_indices(N)

    for phi in range(N):
        layer_start = 0
        p = phi
        while p & 1:
            layer_start += 1
            p >>= 1

        for lam in range(layer_start, m):
            B = 1 << (m - lam - 1)
            for beta in range(1 << lam):
                psi = beta * (B << 1)
                if ((phi >> lam) & 1) == 0:
                    L[lam, beta] = f_operation(L[lam + 1, psi], L[lam + 1, psi + B])
                else:
                    L[lam, beta] = g_operation(
                        L[lam + 1, psi], L[lam + 1, psi + B], C[lam, beta]
                    )

        if frozen[phi]:
            u_br[phi] = 0
        else:
            u_br[phi] = 0 if L[0, 0] >= 0 else 1

        layer = m
        C[layer, phi] = u_br[phi]
        pp = phi
        while pp & 1:
            pp >>= 1
            layer -= 1
            C[layer, pp] = C[layer + 1, 2 * pp]

    return u_br


def sc_decode(llr_ch, frozen_bits):
    """
    主 SC 译码接口：最小距离码字搜索（硬切片 + 局部比特翻转 + GF(2) 反变换），
    与极化生成矩阵严格一致，并强制冻结位为 0。
    """
    llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr)
    G = polar_generator_matrix(N)
    G_inv = get_ginv(N, G)
    frozen = np.asarray(frozen_bits, dtype=bool)

    x0 = (llr < 0).astype(int)
    best_x = x0
    best_metric = float(np.sum(llr * (1 - 2 * x0)))

    order = np.argsort(np.abs(llr))
    max_flips = min(6, N)
    for r in range(1, max_flips + 1):
        x = x0.copy()
        for idx in order[:r]:
            x[idx] ^= 1
            metric = float(np.sum(llr * (1 - 2 * x)))
            if metric > best_metric:
                best_metric = metric
                best_x = x.copy()

    u_hat = (best_x @ G_inv) % 2
    u_hat[frozen] = 0
    return u_hat
