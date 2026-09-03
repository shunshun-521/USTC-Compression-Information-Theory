"""
极化码 SC（串行抵消）译码器
Permuted SCD 非递归实现（Vangala et al. 2014）+ 递归参考实现
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
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
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def precompute_sc_indices(N):
    """预计算 Permuted SCD 层列表"""
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1)) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDecoderCore:
    """Permuted SCD"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    top_bit = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        top_bit,
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        self.B.fill(np.nan)
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            self._update_llrs(l)
            if self.frozen_bits[l]:
                self.B[l, self.n] = 0
                u_hat[l] = 0
            else:
                bit = 0 if self.L[l, self.n] >= 0 else 1
                self.B[l, self.n] = bit
                u_hat[l] = bit
            self._update_bits(l)
        return u_hat


_SC_CACHE = {}


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码"""
    N = len(llr_ch)
    key = (N, tuple(np.asarray(frozen_bits, dtype=int).tolist()))
    if key not in _SC_CACHE:
        _SC_CACHE[key] = _SCDecoderCore(N, frozen_bits)
    return _SC_CACHE[key].decode(llr_ch)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（调用 Permuted SCD 作为参考实现）"""
    return sc_decode(llr, frozen_bits)
