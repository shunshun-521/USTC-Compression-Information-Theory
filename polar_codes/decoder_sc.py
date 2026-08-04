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
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    """精确 log-domain f 运算"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if (x >> i) & 1:
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


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（min-sum，参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)
    half = N // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[:half], llr[half:], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（保留接口兼容性）"""
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        for layer in range(n):
            if (phi >> layer) & 1:
                break
            llr_layer_vec[phi].append(layer)
        if phi % 2 == 0:
            for layer in range(n):
                if (phi >> layer) & 1:
                    bit_layer_vec[phi].append(layer)
        else:
            for layer in range(n):
                if not ((phi >> layer) & 1):
                    bit_layer_vec[phi].append(layer)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDCore:
    """非递归 SC 译码核心（精确 log-domain f，按比特倒序译码）"""

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
                    self.L[j, s + 1] = _upper_llr_exact(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    self.L[j, s + 1] = _lower_llr_exact(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr):
        self.L[:, 0] = llr
        for l in [_bit_reversed_index(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 1 if self.L[l, self.n] < 0 else 0
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    对信道 LLR 做比特倒序置换以匹配编码器的 B_N 置换。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_perm = llr_ch[br]
    return _SCDCore(N, frozen_bits).decode(llr_perm)
