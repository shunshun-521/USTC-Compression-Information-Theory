"""
极化码 SC（串行抵消）译码器
非递归矩阵遍历实现（参考 PolarCodesPython/sc_decoder 思路）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum f（与 f_hf 一致）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    if s1.ndim:
        s1[s1 == 0] = 1
        s2[s2 == 0] = 1
    else:
        if s1 == 0:
            s1 = 1.0
        if s2 == 0:
            s2 = 1.0
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _all_filled(x):
    return not np.any(np.isnan(x))


def _sc_matrix_decode(y_llr, info_indices, frozen_value=0):
    """核心 SC：llr_matrix[0] 为信道 LLR。"""
    y_llr = np.asarray(y_llr, dtype=np.float64)
    N = y_llr.size
    n = int(math.log2(N))
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    def up(pos):
        p0 = pos[0] - 1
        p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
        return [p0, p1, pos[2], pos[3]]

    def get_up_bit(left_bit, right_bit):
        length = left_bit.size
        temp = np.array([(left_bit + right_bit) % 2, right_bit])
        temp.resize((1, 2 * length))
        return temp.ravel()

    def get_left_llr(up_llr):
        length = int(up_llr.size / 2)
        return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])

    def get_right_llr(left_bit, up_llr):
        length = int(left_bit.size)
        return np.array(
            [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
        )

    def get_left_bit(left_llr, pos_idx):
        if pos_idx in info_set:
            return 0 if left_llr >= 0 else 1
        return frozen_value

    def get_right_bit(right_llr, pos_idx):
        if pos_idx in info_set:
            return 0 if right_llr > 0 else 1
        return frozen_value

    while not _all_filled(bit_matrix[n]):
        up_llr = llr_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + 2 ** (position[2] - position[0])]
        span = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + span]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + span]
        right_llr = llr_matrix[position[0] + 1][position[1] + span : position[1] + 2 * span]
        right_bit = bit_matrix[position[0] + 1][position[1] + span : position[1] + 2 * span]

        if _all_filled(up_bit):
            position = up(position)
        else:
            if _all_filled(right_bit):
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][position[1] : position[1] + 2 * span] = up_bit.copy()
            else:
                if _all_filled(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + span
                        right_bit = get_right_bit(right_llr[0], right_bit_pos)
                        bit_matrix[position[0] + 1][position[1] + span : position[1] + 2 * span] = right_bit
                    else:
                        position = rightdown(position)
                else:
                    if _all_filled(left_bit):
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][position[1] + span : position[1] + 2 * span] = right_llr
                    else:
                        if not _all_filled(left_llr):
                            left_llr = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][position[1] : position[1] + span] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = get_left_bit(left_llr[0], left_bit_pos)
                                bit_matrix[position[0] + 1][position[1] : position[1] + span] = left_bit
                            else:
                                position = leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    info_idx = np.where(np.asarray(frozen_bits) == 0)[0]
    return _sc_matrix_decode(llr, info_idx, frozen_value=0)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_nonrecursive(llr_ch, frozen_bits)


# SCL 复用占位（由 decoder_scl 使用 f/g）
def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], None, None


def _update_llr_path(phi, n, P, C, lambda_offset, llr_layer_vec):
    pass


def _update_bit_path(phi, n, C, lambda_offset, bit_layer_vec):
    pass
