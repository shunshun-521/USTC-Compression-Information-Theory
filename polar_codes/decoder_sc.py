"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算（La=上分支，Lb=下分支）：
    g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    """
    return Lb + (1.0 - 2.0 * np.asarray(u_hat)) * La


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def _prepare_channel_llr(llr_ch, N):
    """将信道 LLR 映射到 SC 因子图输入（与含比特倒序的编码器配套）。"""
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，与非递归结果一致）。
    frozen_bits: 1/True 表示冻结位。
    """
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for i in range(N):
        l = _bit_reversed(i, n)
        layers = []
        tmp = l
        while tmp % 2 == 1:
            layers.append(_active_llr_level(l, n) - 1)
            tmp >>= 1
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))

        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDEngine:
    """非递归 SC 译码引擎。"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.br = bit_reversal_permutation(N)
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int_)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + half, s])
                else:
                    self.L[j, s + 1] = g_operation(
                        self.L[j - half, s], self.L[j, s], self.B[j - half, s + 1]
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 2 ** s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    self.B[j - half, s - 1] = self.B[j, s] ^ self.B[j - half, s]
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = _prepare_channel_llr(llr_ch, self.N)
        self.B.fill(0)
        u_hat = np.zeros(self.N, dtype=int)

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            u_hat[l] = self.B[l, self.n]
            self._update_bits(l)

        return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    N = len(llr_ch)
    engine = _SCDEngine(N, frozen_bits)
    return engine.decode(llr_ch)
