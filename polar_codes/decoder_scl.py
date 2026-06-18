"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _all_num,
    _get_left_bit,
    _get_left_llr,
    _get_right_bit,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
    sc_decode,
)


def _crc_poly(crc_length):
    """生成 CRC 多项式系数（GF(2)）。"""
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    poly = [0] * (crc_length + 1)
    for i in loc:
        poly[i] = 1
    return poly[::-1]


def _crc_encode_bits(info_bits, crc_length):
    """GF(2) 多项式除法求 CRC 校验位。"""
    info = [int(b) for b in info_bits]
    poly = _crc_poly(crc_length)
    work = info + [0] * crc_length
    times = len(info)
    for i in range(times):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[i + j] ^= poly[j]
    return np.array(info + work[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    return _crc_encode_bits(info_bits, crc_length)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    if crc_length == 0:
        return True
    bits = list(np.asarray(bits, dtype=int))
    info_len = len(bits) - crc_length
    recoded = _crc_encode_bits(bits[:info_len], crc_length)
    return all(bits[i] == recoded[i] for i in range(len(bits)))


def _get_up_loc(bit_matrix):
    n_layer = bit_matrix.shape[0] - 1
    detect_array = bit_matrix[n_layer]
    detect = -1
    for i in range(len(detect_array)):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n_layer - 1, detect]
    return [n_layer - 1, detect - 1]


def _pm_update(llr_array, bit_array):
    """路径度量更新（hf 方法）。"""
    pm = 0.0
    for i in range(len(llr_array)):
        expected_sign = 1 - 2 * bit_array[i]
        if np.sign(llr_array[i]) != np.sign(expected_sign):
            pm += abs(llr_array[i])
    return pm


def _sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """SC 译码至 split_pos 判决完成。"""
    N = int(bit_matrix.shape[1])
    n = int(np.log2(N))
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n][split_pos] != 0 and bit_matrix[n][split_pos] != 1:
        up_llr = llr_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1]:position[1] + 2 ** (position[2] - position[0])
        ]
        left_llr = llr_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        left_bit = bit_matrix[position[0] + 1][
            position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
        ]
        right_llr = llr_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1):
            position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit):
            position = _up(position)
        else:
            if _all_num(right_bit):
                up_bit = _get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1]:position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if _all_num(right_llr):
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = _get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit):
                        right_llr = _get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1):
                            position[1] + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if not _all_num(left_llr):
                            left_llr = _get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = _get_left_bit(
                                    left_llr, information_pos, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1]:position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = np.where(self.frozen_bits == 0)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        y_llr = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        information_pos = self.information_pos
        frozen_bit = 0
        list_max = self.list_size

        llr_matrix = np.ones((n + 1, N), dtype=np.float64)
        llr_matrix[:] = np.nan
        bit_matrix = np.ones((n + 1, N), dtype=np.float64)
        bit_matrix[:] = np.nan
        llr_matrix[0] = y_llr

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_pos = list(information_pos)
        split_loc = 0
        split_len = len(split_pos)
        l_now = 1

        while split_len - 1 >= split_loc:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                pm_temp = pm_list[i]

                llr_out, bit_out = _sc_stepping_decoder(
                    llr_temp, bit_temp, information_pos, frozen_bit, split_pos[split_loc]
                )

                prev_end = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                cur_end = split_pos[split_loc] + 1
                llr_slice = llr_out[n][prev_end:cur_end]
                bit_slice = bit_out[n][prev_end:cur_end]

                new_llr_list.append(llr_out)
                new_bit_list.append(bit_out)
                new_pm_list.append(pm_temp + _pm_update(llr_slice, bit_slice))

                bit_wrong = bit_out.copy()
                bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
                bit_wrong_slice = bit_wrong[n][prev_end:cur_end]
                new_llr_list.append(llr_out.copy())
                new_bit_list.append(bit_wrong)
                new_pm_list.append(pm_temp + _pm_update(llr_slice, bit_wrong_slice))

            if l_now > list_max // 2:
                order = np.argsort(new_pm_list)[:list_max]
                new_llr_list = [new_llr_list[i] for i in order]
                new_bit_list = [new_bit_list[i] for i in order]
                new_pm_list = [new_pm_list[i] for i in order]

            llr_list = new_llr_list
            bit_list = new_bit_list
            pm_list = new_pm_list
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != N - 1:
            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                pm_temp = pm_list[i]
                llr_out, bit_out = _sc_stepping_decoder(
                    llr_temp, bit_temp, information_pos, frozen_bit, N - 1
                )
                prev_end = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                llr_list[i] = llr_out
                bit_list[i] = bit_out
                pm_list[i] = pm_temp + _pm_update(
                    llr_out[n][prev_end:N], bit_out[n][prev_end:N]
                )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][n].astype(int)
                info_bits = u_cand[information_pos]
                if crc_check(info_bits, self.crc_length):
                    return u_cand, pm_list[idx]
            best_u = bit_list[order[0]][n].astype(int)
        else:
            best_u = bit_list[order[0]][n].astype(int)

        return best_u, best_pm
