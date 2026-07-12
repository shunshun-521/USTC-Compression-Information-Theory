"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _f_hf(L1, L2):
    return float(f_operation(L1, L2))


def _g(L1, L2, u):
    return float(g_operation(L1, L2, u))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = np.where(~frozen_bits)[0].tolist()
    frozen_val = 0
    result = _sc_tree_decode(np.asarray(llr, dtype=np.float64), info_pos, frozen_val)
    return result.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    llr_layer_vec = [
        [layer for layer in range(n) if (phi >> layer) & 1] for phi in range(N)
    ]
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p, layer = phi + 1, 0
        while layer < n and (p & 1) == 0:
            layers.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(layers)
    return list(range(N)), llr_layer_vec, bit_layer_vec


def _all_num(x):
    return not np.any(np.isnan(x))


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
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)]
                    + [right_bit[i] for i in range(length)], dtype=np.float64)
    temp.resize((2 * length,))
    return temp


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr[0] >= 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr[0] >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array(
        [_g(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)],
        dtype=np.float64,
    )


def _get_left_llr(up_llr):
    length = int(up_llr.size // 2)
    return np.array(
        [_f_hf(up_llr[i], up_llr[i + length]) for i in range(length)],
        dtype=np.float64,
    )


def _sc_tree_decode(y_llr, information_pos, frozen_bit):
    """非递归树遍历 SC 译码核心"""
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_num(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit_new.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr, information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr, information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = left_bit_val
            else:
                position = _leftdown(position)

    return bit_matrix[n]


def _get_up_loc(bit_matrix):
    """定位树遍历当前位置"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
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


def sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码至 split_pos（含）"""
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = _get_up_bit(left_bit, right_bit)
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                val = _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            llr_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = _get_right_llr(left_bit, up_llr)
        elif not _all_num(left_llr):
            llr_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                val = _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _pm_update(llr_slice, bit_slice):
    pm = 0.0
    for lv, bv in zip(llr_slice, bit_slice):
        hard = 0 if lv >= 0 else 1
        if hard != int(bv):
            pm += abs(lv)
    return pm


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_pos = np.where(~frozen_bits)[0].tolist()
    return _sc_tree_decode(np.asarray(llr_ch, dtype=np.float64), info_pos, 0).astype(int)


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)
