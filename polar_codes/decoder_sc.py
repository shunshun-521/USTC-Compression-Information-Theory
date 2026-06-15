"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSC 版本（高效实现）
"""
import math

import numpy as np


def _bit_reversed(x, n):
    """单索引比特倒序。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    f 运算（log-domain boxplus，对标 min-sum 近似）。
  """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.shape != ():
        return np.vectorize(_f_scalar)(La, Lb)
    return _f_scalar(La, Lb)


def _f_scalar(l1, l2):
  # min-sum 近似（向量化接口保留）
    return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))


def f_operation_exact(La, Lb):
    """精确 log-domain boxplus f 运算。"""
    return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    La=top, Lb=bottom
    """
    return (1 - 2 * u_hat) * La + Lb


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


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        decode_node(f_operation_exact(llr_node[:half], llr_node[half:]), bit_offset)
        decode_node(
            g_operation(llr_node[:half], llr_node[half:], u_hat[bit_offset:bit_offset + half]),
            bit_offset + half,
        )

    decode_node(llr_ch, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算 PSC 译码的比特倒序处理顺序。"""
    n = int(np.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    return decode_order, n


class _PSCState:
    """PSC 译码内部状态。"""

    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top = self.L[j, s]
                    btm = self.L[j + branch_size, s]
                    self.L[j, s + 1] = f_operation_exact(top, btm)
                else:
                    btm = self.L[j, s]
                    top = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(top, btm, top_bit)

    def update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    PSC 非递归 SC 译码（Permuted Successive Cancellation）。
    信道 LLR 保持自然顺序，按比特倒序依次译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    decode_order, _ = precompute_sc_indices(N)
    state = _PSCState(N, n, llr_ch)

    for l in decode_order:
        state.update_llrs(l)
        if frozen_bits[l]:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        state.update_bits(l)

    return state.B[:, n].astype(int)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test errors: {errors}/100")
