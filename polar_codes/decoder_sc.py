"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）和非递归版本（高效实现，Permuted SCD）
"""
import math
import numpy as np

from encoder import bit_reversed


def logdomain_sum(x, y):
    if x >= y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """精确 f 运算（对数域 box-plus），向量化标量路径"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(lambda a, b: logdomain_sum(a + b, 0.0) - logdomain_sum(a, b))(
        La, Lb
    )


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def upper_llr(l1, l2):
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


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


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    return set(np.where(frozen_bits)[0])


class _SCDCore:
    def __init__(self, N, frozen_set):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = frozen_set
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def set_channel(self, llr_ch):
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
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

    def decode_bits(self):
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(np.int8)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    core = _SCDCore(N, _frozen_indices(frozen_bits))
    core.set_channel(llr_ch)
    return core.decode_bits()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（使用 f/g 与主译码器相同的 LLR 定义）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        u_left = np.zeros(half, dtype=np.int8)
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)
            u_left[i] = u_hat[bit_offset + i]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)
        return None

    order = [bit_reversed(i, int(math.log2(N))) for i in range(N)]
    llr_perm = np.zeros(N)
    inv = np.zeros(N, dtype=int)
    for i, l in enumerate(order):
        inv[l] = i
    for l in range(N):
        llr_perm[order[l]] = llr[l]
    decode_node(llr_perm, 0)
    return u_hat


def precompute_sc_indices(N):
    """兼容 SCL：返回占位结构（SCL 使用 SCD 核心逻辑）"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]
