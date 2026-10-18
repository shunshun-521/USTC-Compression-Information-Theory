"""
极化码 SC（串行抵消）译码器
置换 SC（Vangala et al.）+ 递归参考实现
"""
import numpy as np
import math


def bit_reversed(i, n):
    """n 位比特倒序"""
    rev = 0
    for _ in range(n):
        rev = (rev << 1) | (i & 1)
        i >>= 1
    return rev


def hard_decision(llr):
    return 0 if llr >= 0 else 1


def logdomain_sum(a, b):
    """log(1 + exp(a-b)) + b 的稳定形式用于 box-plus"""
    if a == np.inf:
        return b
    if b == np.inf:
        return a
    if a == -np.inf:
        return b
    if b == -np.inf:
        return a
    if a > b:
        return a + math.log1p(math.exp(b - a))
    return b + math.log1p(math.exp(a - b))


def upper_llr(l1, l2):
    """f 运算（对数域 box-plus）"""
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 运算（对数域）"""
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
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


def f_operation(La, Lb):
    """min-sum f（向量化，供 SCL/BP 复用）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（与 lower_llr 一致）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


class _SCDState:
    """置换 SC 译码内部状态"""

    def __init__(self, N, n, llr_ch, frozen_bits):
        self.N = N
        self.n = n
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(
                        self.L[j, s], self.L[j - branch_size, s], top_bit
                    )

    def update_bits(self, l):
        if l < self.N // 2:
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


def sc_decode(llr_ch, frozen_bits):
    """
    置换 SC 译码主函数。
    信道 LLR 按 BPSK 符号自然顺序输入（不对 LLR 做比特倒序）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    dec = _SCDState(N, n, llr_ch, frozen_bits)

    for i in range(N):
        l = bit_reversed(i, n)
        dec.update_llrs(l)
        if dec.frozen[l]:
            dec.B[l, n] = 0
        else:
            dec.B[l, n] = hard_decision(dec.L[l, n])
        dec.update_bits(l)

    return dec.B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（自然顺序 LLR，与置换版等价性用于短码校验）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """兼容接口：返回占位预计算结构"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
