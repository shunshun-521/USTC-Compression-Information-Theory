"""SC decoder ported from factor-graph stepping algorithm."""
import numpy as np


def f_operation(La, Lb):
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = 1 if sa == 0 else sa
    sb = 1 if sb == 0 else sb
    return sa * sb * min(abs(La), abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_num(x):
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
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp.ravel()


def _get_right_bit(right_llr, frozen_set, right_bit_pos):
    if right_bit_pos in frozen_set:
        return 0
    return 0 if right_llr > 0 else 1


def _get_left_bit(left_llr, frozen_set, left_bit_pos):
    if left_bit_pos in frozen_set:
        return 0
    return 0 if left_llr >= 0 else 1


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], int(left_bit[i])) for i in range(length)]
    )


def _get_left_llr(up_llr):
    length = int(up_llr.size // 2)
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_up_loc(bit_matrix):
    N = int(bit_matrix.shape[1])
    n = int(np.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if detect_array[i] == 1 or detect_array[i] == 0:
            continue
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


def _sc_step_once(llr_matrix, bit_matrix, frozen_set, stop_pos=None):
    """执行 SC 步进直到 bit_matrix[n][stop_pos] 已判决，或全部完成。"""
    N = int(bit_matrix.shape[1])
    n = int(np.log2(N))
    if stop_pos is None:
        stop_pos = N - 1
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, stop_pos] != 0 and bit_matrix[n, stop_pos] != 1:
        up_llr = llr_matrix[position[0], position[1] : position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0], position[1] : position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[
            position[0] + 1, position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[
            position[0] + 1, position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[
            position[0] + 1,
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0]),
        ]
        right_bit = bit_matrix[
            position[0] + 1,
            position[1]
            + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0]),
        ]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            up_bit = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[
                position[0], position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit.copy()
        elif _all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 2 ** (position[2] - position[0] - 1)
                right_bit_val = _get_right_bit(right_llr[0], frozen_set, right_bit_pos)
                bit_matrix[
                    position[0] + 1,
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0]),
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            right_llr_new = _get_right_llr(left_bit.astype(int), up_llr)
            llr_matrix[
                position[0] + 1,
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0]),
            ] = right_llr_new
        elif not _all_num(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[
                position[0] + 1, position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(left_llr[0], frozen_set, left_bit_pos)
                bit_matrix[
                    position[0] + 1, position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def sc_decode_stepping(y_llr, frozen_set):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    _, bit_matrix = _sc_step_once(llr_matrix, bit_matrix, frozen_set, N - 1)
    return bit_matrix[n].astype(int)
