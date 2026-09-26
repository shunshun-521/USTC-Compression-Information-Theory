"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效）
"""
import numpy as np
import math


def bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


def f_operation(La, Lb):
    """min-sum 近似 f（仿真用）"""
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（min-sum），输入信道 LLR（与码字比特顺序一致）"""
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


class _SCDEngine:
    """Permuted successive cancellation decoder（对数域 f/g）"""

    def __init__(self, N, frozen_bits, use_minsum=False):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(frozen_bits)[0])
        self.use_minsum = use_minsum
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def _f(self, l1, l2):
        if self.use_minsum:
            return float(f_operation(np.array([l1]), np.array([l2]))[0])
        return upper_llr(l1, l2)

    def _g(self, l1, l2, b):
        # l1: bottom branch LLR, l2: top branch LLR（与 lower_llr 参数顺序一致）
        if self.use_minsum:
            return float(g_operation(np.array([l2]), np.array([l1]), np.array([b]))[0])
        return lower_llr(l1, l2, b)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = self._f(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = self._g(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode_codeword_layer(self):
        for l in [bit_reversed_index(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def _polar_f_transform(x):
    """对向量 x 施加 F^{\\otimes n}（蝶形，与 Encode.polar_encode 一致，无比特倒序）"""
    u = np.array(x, dtype=np.int8, copy=True)
    N = len(u)
    n = int(np.log2(N))
    block = N
    while block > 1:
        half = block // 2
        for p in range(0, N, block):
            for k in range(half):
                u[p + k] ^= u[p + k + half]
        block = half
    return u.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助结构（与 Permuted SCD 层活跃模式对应）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed_index(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        if l >= N / 2:
            bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])
    lambda_offset = np.arange(n + 1)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _channel_to_core_llr(llr_ch):
    """将信道码字顺序 LLR 映射到 Permuted SCD 使用的 core 顺序"""
    from encoder import bit_reversal_permutation

    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=np.int64)
    for i in range(N):
        inv[br[i]] = i
    return np.asarray(llr_ch, dtype=np.float64)[inv]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Permuted SCD）。
    llr_ch: 与信道发送码字 polar_encode(u) 的比特顺序一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    engine = _SCDEngine(N, frozen_bits, use_minsum=True)
    engine.L[:, 0] = _channel_to_core_llr(llr_ch)
    return engine.decode_codeword_layer()


def sc_decode_logdomain(llr_ch, frozen_bits):
    """对数域 f/g 的 SC（用于无损验证）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    engine = _SCDEngine(N, frozen_bits, use_minsum=False)
    engine.L[:, 0] = _channel_to_core_llr(llr_ch)
    return engine.decode_codeword_layer()
