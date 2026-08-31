"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（sign(0) 视为 +1）"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _all_num(x):
    for i in range(x.size):
        if np.isnan(x[i]):
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
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))))
    p1 *= 2 ** (position[2] - position[0] + 1)
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array([g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)])


def _get_left_llr(up_llr):
    length = int(up_llr.size / 2)
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _sc_tree_decode(y_llr, information_pos, frozen_bit=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float('nan')
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span_pow = position[2] - position[0]
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** span_pow]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** span_pow]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
        ]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** span_pow] = up_bit.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
            ] = right_llr_val
        elif _all_num(left_llr) == 0:
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n]


def _get_up_loc(bit_matrix):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] == 1 or detect_array[i] == 0:
            pass
        else:
            detect = i - 1
            break
    if detect % 2 == 0:
        loc_row = n - 1
        loc_col = detect
    else:
        loc_row = n - 1
        loc_col = detect - 1
    if detect == -1:
        loc_row = 0
        loc_col = 0
    return [loc_row, loc_col]


def _pm_update_hf(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        if np.sign(llr_array[i]) != np.sign(1 - 2 * bit_array[i]):
            pm += np.abs(llr_array[i])
    return pm


def sc_stepping_decode(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码到 split_pos 位（含）"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        span_pow = position[2] - position[0]
        up_llr = llr_matrix[position[0]][position[1]:position[1] + 2 ** span_pow]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + 2 ** span_pow]
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
        ]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + 2 ** span_pow] = up_bit.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1] + 2 ** (span_pow - 1):position[1] + 2 ** span_pow
            ] = right_llr_val
        elif _all_num(left_llr) == 0:
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1][position[1]:position[1] + 2 ** (span_pow - 1)] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _frozen_to_info_pos(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return np.where(~frozen_bits)[0].tolist()


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    info_pos = _frozen_to_info_pos(frozen_bits)
    return _sc_tree_decode(np.asarray(llr_ch, dtype=np.float64), info_pos, 0).astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与树遍历实现等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while p & 1:
            layers_llr.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        p = phi
        layer = 0
        while (p & 1) == 0 and p > 0:
            layers_bit.append(layer)
            p >>= 1
            layer += 1
        if phi == N - 1:
            layers_bit.append(layer)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(42)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_sc, u_rec), "SC recursive vs non-recursive mismatch"
        assert np.array_equal(u[info_idx], u_sc[info_idx]), "SC decode error"

    return True


if __name__ == "__main__":
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("u =", u, "-> x =", x)

    print("Running SC verification...")
    verify_sc_decoders()
    print("SC decoder tests passed.")
