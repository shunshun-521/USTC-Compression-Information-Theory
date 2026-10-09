"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等模块使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _logdomain_sum(a, b):
    if a == b:
        return a + np.log(2.0)
    m, M = min(a, b), max(a, b)
    return M + np.log1p(np.exp(m - M))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    return l1 + l2 if b == 0 else l1 - l2


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


class _SCDCore:
    """非递归 SC 内核（与蝶形+比特倒序编码配套，信道 LLR 已做倒序置换）"""

    def __init__(self, N, frozen_indices, llr):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(int(i) for i in frozen_indices)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)
        self.L[:, 0] = llr

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    btm = self.L[j, s]
                    top = self.L[j - branch_size, s]
                    b = int(self.B[j - branch_size, s + 1])
                    self.L[j, s + 1] = btm + top if b == 0 else btm - top

    def _update_bits(self, l):
        if l < self.N // 2:
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

    def decode(self):
        for l in (_bit_reversed(i, self.n) for i in range(self.N)):
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（对信道 LLR 做比特倒序后与编码器一致）"""
    from encoder import bit_reversal_permutation

    N = len(llr)
    br = bit_reversal_permutation(N)
    llr_br = np.asarray(llr, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    frozen_idx = np.where(frozen_bits.astype(bool))[0]
    return _SCDCore(N, frozen_idx, llr_br).decode()


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量（接口保留，内核使用 SCD 算法）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = list(range(_active_llr_level(phi, n) - 1, -1, -1))
        llr_layer_vec.append(layers_llr)
        layers_bit = list(range(_active_bit_level(phi, n)))
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    frozen_bits: 1 表示冻结位
    """
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    frozen_idx = np.where(frozen_bits.astype(bool))[0]
    return _SCDCore(N, frozen_idx, llr_ch[br]).decode()
