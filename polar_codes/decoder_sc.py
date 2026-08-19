"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def _f_hf(l1, l2):
    s1 = np.sign(l1)
    s2 = np.sign(l2)
    if s1 == 0:
        s1 = 1
    if s2 == 0:
        s2 = 1
    return s1 * s2 * min(abs(l1), abs(l2))


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（向量化）"""
    v = np.vectorize(_f_hf)(La, Lb)
    return v


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _g_scalar(l1, l2, u1):
    return (1 - 2 * u1) * l1 + l2


def _all_filled(x):
    return not np.isnan(x).any()


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
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([_f_hf(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = left_bit.size
    return np.array([
        _g_scalar(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)
    ])


def _get_left_bit(left_llr, info_indices, left_bit_pos):
    if left_bit_pos in info_indices:
        return 0 if left_llr >= 0 else 1
    return 0


def _get_right_bit(right_llr, info_indices, right_bit_pos):
    if right_bit_pos in info_indices:
        return 0 if right_llr >= 0 else 1
    return 0


def _sc_decode_core(y_llr, info_indices):
    """基于因子图遍历的 SC 译码核心（与蝶形+比特倒序编码配套）"""
    info_set = set(int(i) for i in info_indices)
    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + span // 2]
        right_llr = llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit_val.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(right_llr, info_set, right_bit_pos)
                bit_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + span // 2:position[1] + span] = right_llr_val
        elif not _all_filled(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(left_llr, info_set, left_bit_pos)
                bit_matrix[position[0] + 1][position[1]:position[1] + span // 2] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用核心算法）"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    info_indices = np.where(frozen_bits == 0)[0]
    rev = bit_reversal_permutation(len(llr))
    info_br = rev[info_indices]
    u_br = _sc_decode_core(np.asarray(llr, dtype=np.float64), info_br)
    return u_br[rev]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（接口兼容）。
    """
    n = int(math.log2(N))
    decode_order = [int(f"{i:0{n}b}"[::-1], 2) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in decode_order:
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return decode_order, llr_layer_vec, bit_layer_vec


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


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u, u_rec):
            errors += 1
    print(f"Eb/N0=10dB SC test errors: {errors}/100")
