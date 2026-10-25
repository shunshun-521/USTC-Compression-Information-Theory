"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 Permuted SCD（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb；u=1 -> La-Lb"""
    u_hat = np.asarray(u_hat)
    return La + (1 - 2 * u_hat) * Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation_exact(La, Lb):
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def _bit_reversed_index(x, n):
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


def _permute_llr_to_decoder_order(llr_ch):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[br] = np.arange(N)
    return np.asarray(llr_ch, dtype=np.float64)[inv]


class _PermutedSCD:
    """Permuted 非递归 SC 译码器（Vangala et al.）"""

    def __init__(self, N, llr_internal, frozen_indices):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(frozen_indices)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_internal

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation_exact(self.L[j, s], self.L[j + branch_size, s])
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    if top_bit == 0:
                        self.L[j, s + 1] = btm_llr + top_llr
                    elif top_bit == 1:
                        self.L[j, s + 1] = btm_llr - top_llr
                    else:
                        self.L[j, s + 1] = np.nan

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation_exact(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(_permute_llr_to_decoder_order(llr), 0)
    return u_hat


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for i in range(1, n):
        lambda_offset[i] = lambda_offset[i - 1] + 2 ** (i - 1)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(np.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 0:
            psi = phi
            while psi % 2 == 0 and psi > 0:
                layers_bit.append(int(np.log2(psi & -psi)))
                psi //= 2
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    frozen_idx = np.where(frozen_bits.astype(bool))[0]
    llr_internal = _permute_llr_to_decoder_order(llr_ch)
    return _PermutedSCD(len(llr_ch), llr_internal, frozen_idx).decode()
