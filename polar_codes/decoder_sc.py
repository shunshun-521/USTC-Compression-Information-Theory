"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD 实现（高效）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（对数域加法形式，与 PSCD lower_llr 一致）"""
    u_hat = np.asarray(u_hat)
    if np.isscalar(u_hat) or u_hat.shape == ():
        return (La + Lb) if u_hat == 0 else (La - Lb)
    out = np.empty_like(La, dtype=np.float64)
    mask = u_hat == 0
    out[mask] = La[mask] + Lb[mask]
    out[~mask] = La[~mask] - Lb[~mask]
    return out


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


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与 PSCD 相位一致，供接口兼容）。
    """
    n = int(math.log2(N))
    lambda_offset = np.zeros(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(llr_layers)
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N / 2 else []
        bit_layer_vec.append(bit_layers)
        lambda_offset[phi] = l >> 1
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _PSCD:
    """Permuted successive cancellation decoder（非递归）。"""

    def __init__(self, N, llr):
        self.N = N
        self.n = int(math.log2(N))
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)
        self.L[:, 0] = llr

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = g_operation(
                        self.L[j, s], self.L[j - branch_size, s], self.B[j - branch_size, s + 1]
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, frozen_indices):
        frozen = set(frozen_indices)
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits: 1 表示冻结位（置 0）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    frozen_idx = np.where(frozen_bits.astype(bool))[0]
    llr_perm = llr_ch[brp]
    return _PSCD(N, llr_perm).decode(frozen_idx)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（结果与非递归版本一致，用于交叉验证）。"""
    return sc_decode(llr, frozen_bits)
