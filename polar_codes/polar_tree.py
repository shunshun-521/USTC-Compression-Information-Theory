"""极化码译码树公共运算（SC / SCL 共享）"""
import numpy as np


def all_computed(x):
    """数组中是否已无 NaN。"""
    return not np.any(np.isnan(x))


def leftdown(position):
    p0 = position[0] + 1
    p1 = position[1]
    return [p0, p1, position[2], position[3]]


def rightdown(position):
    p0 = position[0] + 1
    p1 = position[1] + 2 ** (position[2] - 1 - position[0])
    return [p0, p1, position[2], position[3]]


def up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
              * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def get_right_bit(right_llr, is_info, frozen_value, right_bit_pos):
    if is_info[right_bit_pos]:
        return 0 if right_llr > 0 else 1
    return frozen_value


def get_left_bit(left_llr, is_info, frozen_value, left_bit_pos):
    if is_info[left_bit_pos]:
        return 0 if left_llr >= 0 else 1
    return frozen_value


def get_right_llr(left_bit, up_llr, g_operation):
    length = int(left_bit.size)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def get_left_llr(up_llr, f_operation):
    length = int(up_llr.size / 2)
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
    )


def get_up_loc(bit_matrix):
    n = int(np.log2(bit_matrix.shape[0] - 1))
    N = bit_matrix.shape[1]
    for i in range(N):
        if bit_matrix[n][i] == 0 or bit_matrix[n][i] == 1:
            loc = [n, i]
            return loc
    return [0, 0]


def init_matrices(N):
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    return llr_matrix, bit_matrix, n


def sc_tree_step(
    llr_matrix,
    bit_matrix,
    position,
    is_info,
    frozen_value,
    f_operation,
    g_operation,
    stop_pos=None,
):
    """
    执行 SC 树遍历，直到 bit_matrix[n][stop_pos] 已判决。
    stop_pos=None 时译码全部比特。
    """
    n = position[2]
    N = position[3]

    while True:
        if stop_pos is not None and bit_matrix[n][stop_pos] in (0, 1):
            break
        if stop_pos is None and all_computed(bit_matrix[n]):
            break

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

        if all_computed(up_bit):
            position = up(position)
        elif all_computed(right_bit):
            up_bit = get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit.copy()
        elif not all_computed(right_llr):
            if all_computed(left_bit):
                right_llr = get_right_llr(left_bit, up_llr, g_operation)
                llr_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = right_llr
            elif not all_computed(left_llr):
                left_llr = get_left_llr(up_llr, f_operation)
                llr_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = left_llr
            elif position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit = get_left_bit(
                    left_llr[0], is_info, frozen_value, left_bit_pos
                )
                bit_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = left_bit
            else:
                position = leftdown(position)
        elif position[0] == position[2] - 1:
            right_bit_pos = position[1] + 1
            right_bit = get_right_bit(
                right_llr[0], is_info, frozen_value, right_bit_pos
            )
            bit_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = right_bit
        else:
            position = rightdown(position)

    return llr_matrix, bit_matrix, position


def frozen_to_info_mask(N, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    is_info = np.zeros(N, dtype=bool)
    is_info[frozen_bits == 0] = True
    return is_info
