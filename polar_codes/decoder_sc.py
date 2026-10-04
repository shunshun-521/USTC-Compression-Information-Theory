"""
极化码 SC（串行抵消）译码器
置换 SC（与 butterfly+比特倒序编码配套），含递归参考实现
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum f（SCL/BP 复用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
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
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    return l1 - l2


def _bit_reversed(x, n):
    r = 0
    for i in range(n):
        if x & (1 << i):
            r |= 1 << (n - 1 - i)
    return r


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


class PermutedSCD:
    """置换串行抵消译码器（非递归）"""

    def __init__(self, N, frozen_indices):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(int(i) for i in frozen_indices)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
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
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_perm):
        self.L[:, 0] = llr_perm
        for l in self.decode_order:
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def precompute_sc_indices(N):
    """SCL 兼容占位"""
    n = int(np.log2(N))
    return list(range(n + 1)), [[] for _ in range(N)], [[0] for _ in range(N)]


def sc_decode_recursive(llr, frozen_bits):
    """递归 min-sum SC（参考，与主实现可能略有数值差异）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen = np.asarray(frozen_bits, dtype=int).astype(bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    br = bit_reversal_permutation(N)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr[br], 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码：输入信道 LLR（自然码字顺序），输出源向量 u_hat。
    frozen_bits: 1=冻结，0=信息
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    frozen_idx = np.where(np.asarray(frozen_bits, dtype=int).astype(bool))[0]
    dec = PermutedSCD(N, frozen_idx)
    return dec.decode(llr_ch[br])
