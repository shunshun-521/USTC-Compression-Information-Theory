"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Permuted-SC）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（b=0/1 对数域：b=0 为 La+Lb，b=1 为 La-Lb）。"""
    u = np.asarray(u_hat, dtype=np.float64)
    return np.where(u < 0.5, La + Lb, La - Lb)


def _log_sum_exp(a, b):
    """log(exp(a)+exp(b)) 稳定实现。"""
    m = np.maximum(a, b)
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def upper_llr_exact(l1, l2):
    """精确 f 运算（对数域，标量）。"""
    return _log_sum_exp(l1 + l2, 0.0) - _log_sum_exp(l1, l2)


def lower_llr_exact(l1, l2, b):
    """精确 g 运算（对数域，标量）；l1 为下支路，l2 为上支路。"""
    if b == 0:
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    """MSB 起第一个 0 的层数（Permuted-SC）。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """MSB 起第一个 1 的层数。"""
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
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
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
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    br = bit_reversal_permutation(N)
    for phi in range(N):
        l = br[phi]
        start = n - active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            end = n - active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 Permuted-SC 译码（与 encoder.polar_encode 配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    br = bit_reversal_permutation(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    frozen_idx = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = br[i]
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top = L[j, s]
                    btm = L[j + branch_size, s]
                    L[j, s + 1] = upper_llr_exact(top, btm)
                else:
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = lower_llr_exact(btm, top, top_bit)

        if l in frozen_idx:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                        B[j, s - 1] = B[j, s]

    u_hat = B[:, n].astype(int)
    return u_hat
