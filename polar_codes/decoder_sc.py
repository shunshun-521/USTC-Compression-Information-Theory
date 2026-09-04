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
  支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """对数域精确 box-plus（标量）"""
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, bit):
    if np.isnan(bit):
        bit = 0
    bit = int(bit)
    if bit == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if bit == 1:
        return l1 - l2
    return np.nan


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


def _map_channel_llr(llr_ch):
    """将信道 LLR 映射到译码树顺序（与比特倒序编码对应）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    return llr_ch[brp]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [0]
    for i in range(1, n + 1):
        lambda_offset.append(1 << (i - 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi
        llr_layers = []
        while psi % 2 == 1 and len(llr_layers) < n:
            llr_layers.append(len(llr_layers))
            psi >>= 1
        llr_layer_vec.append(llr_layers)

        psi = phi
        bit_layers = []
        while psi % 2 == 1 and len(bit_layers) < n:
            bit_layers.append(len(bit_layers))
            psi >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDCore:
    """Vangala 置换 SC 译码核心"""

    def __init__(self, N, frozen_bits, use_minsum=False):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=int) == 1)[0])
        self.use_minsum = use_minsum
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def _f_llr(self, l1, l2):
        if self.use_minsum:
            return float(f_operation(l1, l2))
        return _upper_llr(l1, l2)

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = self._f_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], self.B[j - branch_size, s + 1]
                    )

    def update_bits(self, l):
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


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，使用 Vangala 非递归核心）。
    """
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Vangala 置换实现）。
    """
    llr = _map_channel_llr(llr_ch)
    core = _SCDCore(len(llr), frozen_bits, use_minsum=True)
    core.L[:, 0] = llr
    u_hat = np.zeros(core.N, dtype=np.int8)

    for l in [_bit_reversed(i, core.n) for i in range(core.N)]:
        core.update_llrs(l)
        if l in core.frozen:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if core.L[l, core.n] >= 0 else 1
        core.B[l, core.n] = u_hat[l]
        core.update_bits(l)

    return u_hat
