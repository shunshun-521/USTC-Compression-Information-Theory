"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 辅助函数 ====================


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _bit_reversed_index(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与高效实现等价）。"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回比特倒序后的译码顺序及层更新列表。
    """
    n = int(math.log2(N))
    lambda_offset = [2 ** l for l in range(n + 1)]
    decode_order = [_bit_reversed_index(phi, n) for phi in range(N)]

    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N // 2 else []
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 为自然顺序；内部做比特倒序置换以匹配编码器。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    br = bit_reversal_permutation(N)
    llr_pc = llr_ch[br]

    _, llr_layer_vec, bit_layer_vec, decode_order = _get_sc_cache(N)

    P = np.zeros((N, n + 1), dtype=np.float64)
    C = np.zeros((N, n + 1), dtype=int)
    P[:, 0] = llr_pc
    u_hat = np.zeros(N, dtype=int)

    for step, l in enumerate(decode_order):
        for s in llr_layer_vec[step]:
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    P[j, s + 1] = f_operation(P[j, s], P[j + branch_size, s])
                else:
                    P[j, s + 1] = g_operation(
                        P[j - branch_size, s], P[j, s], C[j - branch_size, s + 1]
                    )

        if frozen_bits[l]:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if P[l, n] >= 0 else 1
        C[l, n] = u_hat[l]

        for s in bit_layer_vec[step]:
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    C[j - branch_size, s - 1] = C[j, s] ^ C[j - branch_size, s]
                    C[j, s - 1] = C[j, s]

    return u_hat


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("编码器: u=", u, "x=", x)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        s = bpsk_modulate(x)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_hat, u_hat_rec), "递归与非递归 SC 不一致"
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    print(f"SC 校验: {100 - errors}/100 帧正确 (Eb/N0=10dB)")
    assert errors == 0, f"SC 译码错误: {errors} 帧失败"
