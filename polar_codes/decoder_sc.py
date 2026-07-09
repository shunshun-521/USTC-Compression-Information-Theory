"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSC 版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

LLR_CLIP = 30.0


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    out = np.where(np.abs(La) < 1e-12, Lb, out)
    out = np.where(np.abs(Lb) < 1e-12, La, out)
    return out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return np.where(x > y, x + np.log1p(np.exp(y - x)), y + np.log1p(np.exp(x - y)))


def _cn_op(x, y):
    """f 运算（对数域 box-plus，向量化）"""
    x = np.clip(np.asarray(x, dtype=np.float64), -LLR_CLIP, LLR_CLIP)
    y = np.clip(np.asarray(y, dtype=np.float64), -LLR_CLIP, LLR_CLIP)
    return _logdomain_sum(x + y, 0.0) - _logdomain_sum(x, y)


def _vn_op(x, y, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * x + y


def _bit_reversed(x, n):
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


def _prepare_llr(llr_ch):
    """对信道 LLR 做比特倒序，与含 B_N 的编码器对齐"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def _frozen_set(frozen_bits):
    return set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（Sionna 风格，含部分和重编码）。
    """
    llr = _prepare_llr(llr_ch)
    frozen = np.asarray(frozen_bits, dtype=int)

    def decode(llr_node, fr):
        n = len(llr_node)
        if n == 1:
            u = 0 if fr[0] else (1 if llr_node[0] < 0 else 0)
            return np.array([u], dtype=int), np.array([u], dtype=int)
        half = n // 2
        l1, l2 = llr_node[:half], llr_node[half:]
        u1, u1_up = decode(_cn_op(l1, l2), fr[:half])
        u2, u2_up = decode(_vn_op(l1, l2, u1_up), fr[half:])
        u_hat = np.concatenate([u1, u2])
        u1_up = np.bitwise_xor(u1_up.astype(int), u2_up.astype(int))
        u_up = np.concatenate([u1_up, u2_up.astype(int)])
        return u_hat, u_up

    u_hat, _ = decode(llr, frozen)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    返回：lambda_offset, llr_layer_vec, bit_layer_vec
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        for l in range(n):
            if (p & 1) == 0:
                layers_llr.append(l)
            p >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        for l in range(n):
            if (phi >> l) & 1:
                layers_bit.append(l)
        bit_layer_vec.append(layers_bit)

        lambda_offset[phi] = phi >> len(layers_llr)

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PSCDecoder:
    """非递归 PSC 译码器状态"""

    def __init__(self, N):
        self.N = N
        self.n = int(math.log2(N))
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def decode(self, llr_ch, frozen_bits):
        frozen = _frozen_set(frozen_bits)
        self.L[:, 0] = _prepare_llr(llr_ch)
        self.B[:] = np.nan
        u_hat = np.zeros(self.N, dtype=int)

        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            for s in range(self.n - _active_llr_level(l, self.n), self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        self.L[j, s + 1] = _cn_op(self.L[j, s], self.L[j + branch_size, s])
                    else:
                        top_bit = self.B[j - branch_size, s + 1]
                        self.L[j, s + 1] = _vn_op(
                            self.L[j - branch_size, s],
                            self.L[j, s],
                            int(top_bit),
                        )
            if l in frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            u_hat[l] = int(self.B[l, self.n])
            if l >= self.N // 2:
                for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                    block_size = 2 ** s
                    branch_size = block_size // 2
                    for j in range(l, -1, -block_size):
                        if j % block_size >= branch_size:
                            self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                                self.B[j - branch_size, s]
                            )
                            self.B[j, s - 1] = self.B[j, s]
        return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（递归实现，稳定可靠）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
