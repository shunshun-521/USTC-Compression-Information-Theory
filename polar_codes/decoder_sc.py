"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，参考 phased SC）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（硬件友好）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(row):
    return not np.any(np.isnan(row))


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_block(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_block(llr_right, bit_offset + half)

    decode_block(llr, 0)
    return u_hat


# ==================== 非递归 phased SC 译码 ====================


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


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)],
        dtype=np.float64,
    )


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)],
        dtype=np.float64,
    )


def _get_up_bit(left_bit, right_bit):
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * len(left_bit)))
    return temp.flatten()


def _get_left_bit(left_llr, information_pos, frozen_val, pos):
    if pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_val


def _get_right_bit(right_llr, information_pos, frozen_val, pos):
    if pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_val


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（phased LLR / bit 矩阵）。
    frozen_bits: 1/True 表示冻结位（置 frozen_val=0）。
    """
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits)
    N = len(y_llr)
    n = int(math.log2(N))
    information_pos = set(np.where(frozen_bits == 0)[0].tolist())
    frozen_val = 0

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
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

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_new
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        rb = _get_right_bit(
                            right_llr[0], information_pos, frozen_val, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = rb
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr_new
                    else:
                        if not _all_filled(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                lb = _get_left_bit(
                                    left_llr[0], information_pos, frozen_val, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = lb
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_val, split_pos):
    """运行 phased SC 直至完成 split_pos 处比特判决。"""
    N = int(bit_matrix.shape[1])
    n = int(math.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]
    information_pos = set(information_pos)

    while not (
        bit_matrix[n][split_pos] == 0 or bit_matrix[n][split_pos] == 1
    ):
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

        if _all_filled(up_bit):
            position = _up(position)
        else:
            if _all_filled(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit_new
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        rb = _get_right_bit(
                            right_llr[0], information_pos, frozen_val, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = rb
                    else:
                        position = _rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1]
                            + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr_new
                    else:
                        if not _all_filled(left_llr):
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1]
                                + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                lb = _get_left_bit(
                                    left_llr[0], information_pos, frozen_val, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1]
                                    + 2 ** (position[2] - position[0] - 1)
                                ] = lb
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _get_up_loc(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if not (detect_array[i] == 0 or detect_array[i] == 1):
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


def _pm_update_segment(llr_array, bit_array):
    pm = 0.0
    for i in range(len(llr_array)):
        hard = 0.0 if llr_array[i] >= 0 else 1.0
        if bit_array[i] != hard:
            pm += abs(llr_array[i])
    return pm


def precompute_sc_indices(N):
    """保留接口：预计算向量（供 SCL 等模块使用）。"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        if phi > 0:
            psi = phi
            while psi % 2 == 1:
                layers_llr.append(int(math.log2(psi & -psi)))
                psi >>= 1
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        psi = phi + 1
        while psi % 2 == 0 and psi <= N:
            layers_bit.append(int(math.log2(psi & -psi)) - 1)
            psi >>= 1
        bit_layer_vec.append(layers_bit)
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
