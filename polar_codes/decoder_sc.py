"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala 等置换 SC 结构）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 等模块调用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    m = max(x, y)
    return m + np.log1p(np.exp(-abs(x - y)))


def _upper_llr_log(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_log(l1, l2, b):
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息（层数与比特倒序译码顺序）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [_bit_reversed_index(i, n) for i in range(N)]
    for phi in decode_order:
        layers = list(range(n - _active_llr_level(phi, n), n))
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def _prepare_channel_llr(llr_ch, N):
    """编码含比特倒序时，将信道 LLR 映射到 SC 因子图顺序。"""
    from encoder import bit_reversal_permutation

    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（对数域 f，信道 LLR 已对齐因子图）"""
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
        llr_left = np.array(
            [_upper_llr_log(llr_node[i], llr_node[i + half]) for i in range(half)]
        )
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def _update_llrs_scd(L, B, l, n):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = int(2 ** (s + 1))
        branch_size = int(block_size / 2)
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr_log(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = _lower_llr_log(
                    L[j, s], L[j - branch_size, s], int(top_bit)
                )


def _update_bits_scd(B, l, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = int(2 ** s)
        branch_size = int(block_size / 2)
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归置换 SC 译码。
    llr_ch: 与发送码字 x 的自然顺序一致的信道 LLR。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = _prepare_channel_llr(llr_ch, N)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr

    for i in range(N):
        l = _bit_reversed_index(i, n)
        _update_llrs_scd(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits_scd(B, l, n)

    return B[:, n].astype(int)


def verify_sc_decoders(N=64, K=32, num_frames=100, seed=0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(seed)
    sigma = 0.05
    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        llr_int = _prepare_channel_llr(llr, N)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            raise AssertionError("SC decode error at high SNR")
    return True
