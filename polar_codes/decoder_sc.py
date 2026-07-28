"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于分层 L/B 矩阵）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  支持向量化（La, Lb 为同形状 numpy 数组）
    """
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _f_boxplus(l1, l2):
    """精确 log-domain f 运算（标量）。"""
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _g_boxplus(l1, l2, b):
    """精确 log-domain g 运算（标量）。"""
    return l1 + l2 if b == 0 else l1 - l2


def _bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    参数：
        llr: 长度 N 的信道 LLR 数组
        frozen_bits: 长度 N 的 bool/int 数组，非零表示冻结位
    返回：
        u_hat: 长度 N 的估计源序列
    """
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
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

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    br = bit_reversal_permutation(N)
    decode_node(np.asarray(llr, dtype=np.float64)[br], 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if ((phi >> layer) & 1) == 0:
                llr_layers = list(range(layer, n))
                break

        bit_layers = [layer for layer in range(n) if ((phi >> layer) & 1) == 1]

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_scd(llr, frozen_indices, n, use_minsum=False):
    """
    基于分层 L/B 矩阵的非递归 SC 译码内核。
    llr: 已按比特倒序置换后的信道 LLR。
    frozen_indices: 冻结位索引数组。
    """
    N = 2 ** n
    frozen_set = set(int(i) for i in frozen_indices)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    f_fn = (lambda a, b: f_operation(a, b)) if use_minsum else _f_boxplus
    g_fn = (lambda a, b, u: g_operation(a, b, u)) if use_minsum else _g_boxplus

    for l in [_bit_reversed_int(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    if use_minsum:
                        L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    if use_minsum:
                        L[j, s + 1] = g_fn(L[j - branch_size, s], L[j, s], top_bit)
                    else:
                        L[j, s + 1] = g_fn(L[j - branch_size, s], L[j, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。

    编码器输出 x = butterfly(u) 再经比特倒序置换，因此译码前对信道 LLR
    施加相同的比特倒序置换以与因子图叶节点对齐。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)

    br = bit_reversal_permutation(N)
    frozen_indices = np.where(frozen_bits)[0]
    llr_dec = llr_ch[br]

    return _sc_decode_scd(llr_dec, frozen_indices, n, use_minsum=True)
