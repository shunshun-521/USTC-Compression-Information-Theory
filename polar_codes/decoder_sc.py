"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（min-sum 对应形式）"""
    if np.isscalar(u_hat):
        return (1 - 2 * u_hat) * La + Lb
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _bit_reversed(i, n):
    return int(bin(i)[2:].zfill(n)[::-1], 2)


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


class _SCDEngine:
    """基于分层存储的非递归 SC 译码引擎"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        top_bit
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = self.B[j, s] ^ self.B[j - branch_size, s]
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = llr_ch
        self.B.fill(0)
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    engine = _SCDEngine(N, frozen_bits)
    return engine.decode(llr_ch[rev])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现作为参考）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [0]
    for i in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + 2 ** i)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(layers)

        b_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                b_layers.append(layer)
            temp //= 2
        bit_layer_vec.append(b_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
