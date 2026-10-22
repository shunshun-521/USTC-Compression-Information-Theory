"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）与非递归 Vangala 置换 SC（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（La 为上分支 LLR，Lb 为下分支 LLR）"""
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed(x, n):
    """比特倒序索引"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """二进制展开中第一个 1 的位置（自高位计）"""
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
    """二进制展开中第一个 0 的位置"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def upper_llr(l1, l2):
    return f_operation(l1, l2)


def lower_llr(l1, l2, b):
    """l1: 下分支 LLR，l2: 上分支 LLR"""
    if b == 0:
        return l1 + l2
    return l1 - l2


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（按自然索引顺序，供对照）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def rec(node_llr, offset, length):
        if length == 1:
            if frozen_bits[offset]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if node_llr[0] >= 0 else 1
            return
        h = length // 2
        llr_l = f_operation(node_llr[:h], node_llr[h:])
        rec(llr_l, offset, h)
        u_l = u_hat[offset : offset + h]
        llr_r = g_operation(node_llr[:h], node_llr[h:], u_l)
        rec(llr_r, offset + h, h)

    rec(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算层索引（报告用）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layer_vec.append(list(range(n - active_llr_level(phi, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDState:
    def __init__(self, N, frozen_bits, llr_ch):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

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
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        self.B[j, s] ^ self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Vangala 置换顺序，与极化编码 x=uG 配套）。
    信道 LLR 按发送码字顺序填入，无需额外倒序。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    state = _SCDState(N, frozen_bits, llr_ch)
    for i in range(N):
        l = bit_reversed(i, n)
        state.update_llrs(l)
        if frozen_bits[i]:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        state.update_bits(l)

    decoded = state.B[:, n].astype(int)
    # 倒序译码顺序 -> 自然信息位索引
    out = np.zeros(N, dtype=int)
    for i in range(N):
        out[i] = decoded[bit_reversed(i, n)]
    return out
