"""
极化码 SCL / CA-SCL 译码器（基于因子图步进 SC，与 sc_decode 配套）
"""
import numpy as np
from decoder_sc import sc_decode
from _sc_func import (
    all_num,
    leftdown,
    rightdown,
    up,
    get_up_bit,
    get_right_bit,
    get_right_llr,
    get_left_bit,
    get_left_llr,
    get_up_loc,
    get_pm_update,
)


def _crc8_mod(info_bits):
    poly = 0x07
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ (poly << 1)) & 0x1FF
            else:
                reg = (reg << 1) & 0x1FF
    return (reg >> 1) & 0xFF


def _crc16_mod(info_bits):
    poly = 0x8005
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        rem = _crc8_mod(info_bits)
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        rem = _crc16_mod(info_bits)
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    payload = bits[:-crc_length]
    return np.array_equal(bits, crc_encode(payload, crc_length))


def sc_stepping_decoder(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    N = int(bit_matrix[0].size)
    n = int(np.log2(N))
    loc = get_up_loc(bit_matrix)
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
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]
        right_bit = bit_matrix[position[0] + 1][
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
            + 2 ** (position[2] - position[0])
        ]

        if all_num(up_bit) == 1:
            position = up(position)
        else:
            if all_num(right_bit) == 1:
                up_bit = get_up_bit(left_bit, right_bit)
                bit_matrix[position[0]][
                    position[1] : position[1] + 2 ** (position[2] - position[0])
                ] = up_bit.copy()
            else:
                if all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_pos = position[1] + 1
                        right_bit = get_right_bit(
                            right_llr, information_pos, frozen_bit, right_bit_pos
                        )
                        bit_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_bit
                    else:
                        position = rightdown(position)
                else:
                    if all_num(left_bit) == 1:
                        right_llr = get_right_llr(left_bit, up_llr)
                        llr_matrix[position[0] + 1][
                            position[1] + 2 ** (position[2] - position[0] - 1) : position[1]
                            + 2 ** (position[2] - position[0])
                        ] = right_llr
                    else:
                        if all_num(left_llr) == 0:
                            left_llr = get_left_llr(up_llr)
                            llr_matrix[position[0] + 1][
                                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                            ] = left_llr
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_pos = position[1]
                                left_bit = get_left_bit(
                                    left_llr, information_pos, frozen_bit, left_bit_pos
                                )
                                bit_matrix[position[0] + 1][
                                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                                ] = left_bit
                            else:
                                position = leftdown(position)
    return llr_matrix, bit_matrix


def scl_decode(y_llr, information_pos, list_size, crc_length=0):
    N = y_llr.size
    n = int(np.log2(N))
    frozen_bit = 0
    split_pos = list(information_pos)
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    llr_list = [llr_matrix]
    bit_list = [bit_matrix]
    pm_list = [0.0]
    split_loc = 0
    split_len = len(split_pos)
    l_now = 1
    pm_method = "l"

    while split_len - 1 >= split_loc:
        for i in range(l_now):
            llr_m = llr_list[i]
            bit_m = bit_list[i]
            pm_temp = pm_list[i]
            llr_m, bit_m = sc_stepping_decoder(
                llr_m, bit_m, information_pos, frozen_bit, split_pos[split_loc]
            )
            llr_list[i] = llr_m
            bit_list[i] = bit_m
            prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
            cur = split_pos[split_loc] + 1
            pm_list[i] = pm_temp + get_pm_update(
                llr_m[n][prev:cur], bit_m[n][prev:cur], pm_method
            )
            llr_list.append(llr_m.copy())
            bit_wrong = bit_m.copy()
            bit_wrong[n][split_pos[split_loc]] = 1 - bit_wrong[n][split_pos[split_loc]]
            bit_list.append(bit_wrong)
            pm_list.append(
                pm_temp
                + get_pm_update(llr_m[n][prev:cur], bit_wrong[n][prev:cur], pm_method)
            )

        if l_now > list_size // 2:
            order = np.argsort(pm_list)
            keep = order[:list_size]
            pm_list = [pm_list[i] for i in keep]
            llr_list = [llr_list[i] for i in keep]
            bit_list = [bit_list[i] for i in keep]
        l_now = len(pm_list)
        split_loc += 1

    if split_pos[-1] != N - 1:
        for i in range(l_now):
            llr_m = llr_list[i]
            bit_m = bit_list[i]
            pm_temp = pm_list[i]
            llr_m, bit_m = sc_stepping_decoder(
                llr_m, bit_m, information_pos, frozen_bit, N - 1
            )
            llr_list[i] = llr_m
            bit_list[i] = bit_m
            prev = split_pos[split_loc - 1] + 1
            pm_list[i] = pm_temp + get_pm_update(
                llr_m[n][prev:N], bit_m[n][prev:N], pm_method
            )

    order = np.argsort(pm_list)
    best_u = None
    best_pm = pm_list[order[0]]
    info_idx = np.array(information_pos, dtype=int)

    for idx in order:
        u_d = bit_list[idx][n].astype(int)
        if crc_length > 0:
            payload = u_d[info_idx]
            if crc_check(payload, crc_length):
                best_u = u_d
                best_pm = pm_list[idx]
                break
        elif best_u is None:
            best_u = u_d
            best_pm = pm_list[idx]
    if best_u is None:
        best_u = bit_list[order[0]][n].astype(int)
    return best_u, float(best_pm)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_pos = np.where(~self.frozen_bits)[0].tolist()

    def decode(self, llr_ch):
        if self.L <= 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        u, pm = scl_decode(
            np.asarray(llr_ch, dtype=np.float64),
            self.info_pos,
            self.L,
            self.crc_length,
        )
        return u, pm
