"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _bit_reversed(x, n):
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
    精确 box-plus f 运算（对数域实现）。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    return np.vectorize(
        lambda a, b: _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)
    )(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _prepare_llr(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    return llr_ch[bit_reversal_permutation(N)]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = _prepare_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, frozen_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                u_hat[bit_offset] = 0
            else:
                u_hat[bit_offset] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, frozen_node[:half], bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, frozen_node[half:], bit_offset + half)

    decode_node(llr, frozen_bits, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SCD 结构）。
    """
    llr_ch = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    llr = np.zeros((N, n + 1), dtype=np.float64)
    bits = np.zeros((N, n + 1), dtype=int)
    llr[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        leaf = _bit_reversed(phi, n)
        for stage in range(n - _active_llr_level(leaf, n), n):
            block_size = 1 << (stage + 1)
            branch_size = block_size // 2
            for j in range(leaf, N, block_size):
                if j % block_size < branch_size:
                    llr[j, stage + 1] = f_operation(llr[j, stage], llr[j + branch_size, stage])
                else:
                    llr[j, stage + 1] = g_operation(
                        llr[j, stage],
                        llr[j - branch_size, stage],
                        bits[j - branch_size, stage + 1],
                    )

        if leaf in frozen_set:
            bits[leaf, n] = 0
        else:
            bits[leaf, n] = 0 if llr[leaf, n] >= 0 else 1
        u_hat[leaf] = bits[leaf, n]

        if leaf < N // 2:
            continue

        for stage in range(n, n - _active_bit_level(leaf, n), -1):
            block_size = 1 << stage
            branch_size = block_size // 2
            for j in range(leaf, -1, -block_size):
                if j % block_size >= branch_size:
                    bits[j - branch_size, stage - 1] = bits[j, stage] ^ bits[j - branch_size, stage]
                    bits[j, stage - 1] = bits[j, stage]

    return u_hat


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码错误帧数: {errors}"
    print("SC decoder tests passed")
