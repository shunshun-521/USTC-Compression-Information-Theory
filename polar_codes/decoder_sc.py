"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，polarcodes 风格）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    return Lb + (1.0 - 2.0 * u_hat) * La


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation_exact(La, Lb):
    """精确 box-plus f 运算（对数域）"""
    if np.isscalar(La):
        if La == np.inf and Lb != np.inf:
            return Lb
        if La != np.inf and Lb == np.inf:
            return La
        if La == np.inf and Lb == np.inf:
            return np.inf
        return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    out = np.empty_like(La, dtype=np.float64)
    for idx in range(La.size):
        out[idx] = f_operation_exact(La[idx], Lb[idx])
    return out


def g_operation_exact(La, Lb, u_hat):
    """精确 g 运算"""
    if u_hat == 0:
        if La == np.inf or Lb == np.inf:
            return np.inf
        return La + Lb
    return La - Lb


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=np.int8)
        return np.array([0 if llr[0] >= 0 else 1], dtype=np.int8)

    half = N // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[:half], llr[half:], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi & 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi < N - 1:
            layer = 0
            while (phi & (1 << layer)) == 0 and layer < n:
                bit_layers.append(layer)
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_core(llr, frozen_bits, use_exact=False):
    """
    非递归 SC 译码核心（polarcodes SCD 算法）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits.astype(bool))[0])

    if use_exact:
        upper = f_operation_exact
        lower = g_operation_exact
    else:
        upper = f_operation
        lower = lambda btm, top, bit: g_operation(top, btm, bit)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = upper(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = lower(btm_llr, top_llr, top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 按发送顺序输入，内部无需额外置换。
    """
    return sc_decode_core(np.asarray(llr_ch, dtype=np.float64), frozen_bits)


def sc_decode_with_precompute(llr_ch, frozen_bits, precomputed=None):
    return sc_decode(llr_ch, frozen_bits)
