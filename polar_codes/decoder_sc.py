"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与 Permuted SCD 非递归实现（主路径）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（用于递归参考实现）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    r = 0
    for b in range(n):
        if (i >> b) & 1:
            r |= 1 << (n - 1 - b)
    return r


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _active_llr_level(i, n):
    """第一个为 1 的比特位置（从 MSB 起），与 mcba1n 一致。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """第一个为 0 的比特位置（从 MSB 起）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


class _PermutedSCD:
    """Permuted successive cancellation decoder（Vangala et al.）。"""

    def __init__(self, N, frozen_bits, llr_ch):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0].tolist())
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    def decode(self):
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def _channel_llr_for_decoder(llr_ch, N):
    """polar_encode 含比特倒序时，将信道 LLR 映射到蝶形域顺序。"""
    inv = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码。"""
    N = len(llr_ch)
    llr = _channel_llr_for_decoder(llr_ch, N)
    dec = _PermutedSCD(N, frozen_bits, llr)
    return dec.decode()


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC（min-sum），与 Permuted SCD 在多数情形下一致，保留作参考。"""
    N = len(llr_ch)
    llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, idx_start):
        sz = len(llr_node)
        if sz == 1:
            i = idx_start
            u_hat[i] = 0 if frozen_bits[i] else (0 if llr_node[0] >= 0 else 1)
            return
        half = sz // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, idx_start)
        u_left = u_hat[idx_start : idx_start + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, idx_start + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """保留接口：预计算向量（文档兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    offset = 0
    for layer in range(n + 1):
        lambda_offset[layer] = offset
        offset += 1 << (n - layer)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        llr_layers = []
        for layer in range(n):
            if (p & 1) == 0:
                llr_layers.append(layer)
            p >>= 1
        llr_layer_vec.append(llr_layers)
        p = phi
        bit_layers = []
        for layer in range(n):
            if (p & 1) == 1:
                bit_layers.append(layer)
            p >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
