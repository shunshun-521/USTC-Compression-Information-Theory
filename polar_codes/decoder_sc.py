"""
极化码 SC（串行抵消）译码器
提供树遍历版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _all_num(x):
    for v in x:
        if np.isnan(v):
            return 0
    return 1


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _sc_tree_decode(llr, information_pos, frozen_bit=0):
    """树遍历 SC 译码核心。"""
    N = llr.size
    n = int(math.log2(N))
    information_pos = np.asarray(information_pos, dtype=int)

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span_pow = position[2] - position[0]
        half_pow = span_pow - 1
        up_llr = llr_matrix[position[0]][position[1] : position[1] + 2**span_pow]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + 2**span_pow]
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + 2**half_pow]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + 2**half_pow]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2**half_pow : position[1] + 2**span_pow
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2**half_pow : position[1] + 2**span_pow
        ]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + 2**span_pow] = up_bit_new.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit = _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1] + 2**half_pow : position[1] + 2**span_pow
                ] = bit
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1] + 2**half_pow : position[1] + 2**span_pow
            ] = right_llr_new
        elif _all_num(left_llr) == 0:
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + 2**half_pow] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit = _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][position[1] : position[1] + 2**half_pow] = bit
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """树遍历 SC 译码（参考实现）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = np.where(~frozen_bits)[0]
    return _sc_tree_decode(llr, information_pos, frozen_bit=0)


def _update_llrs(l, L, B, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                top_llr = L[j - branch_size, s]
                btm_llr = L[j, s]
                L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)


def _update_bits(l, B, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）。"""
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    llr_layer_vec = [_active_llr_level(br[phi], n) for phi in range(N)]
    bit_layer_vec = [_active_bit_level(br[phi], n) for phi in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（比特倒序信道顺序）。"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(l, L, B, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(l, B, n, N)

    u_hat = B[:, n].astype(int)
    u_hat[frozen_bits] = 0
    return u_hat


def verify_sc_decoder(num_frames=100, N=64, K=32, eb_n0_db=12.0):
    """极低噪声下 SC 译码应全部正确。"""
    from construction import ga_construction
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
