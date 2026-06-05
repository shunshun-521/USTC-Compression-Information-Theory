"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _all_filled,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
    _leftdown,
    _rightdown,
    _up,
)

_CRC_POLY_TAPS = {
    8: [8, 2, 1, 0],
    16: [16, 15, 2, 0],
}


def _crc_poly_bits(crc_length):
    taps = _CRC_POLY_TAPS[crc_length]
    poly = [0] * (crc_length + 1)
    for tap in taps:
        poly[tap] = 1
    return poly[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（多项式长除法）。"""
    info_bits = np.asarray(info_bits, dtype=int).tolist()
    poly = _crc_poly_bits(crc_length)
    work = info_bits + [0] * crc_length
    q = []
    for i in range(len(info_bits)):
        if work[i] == 1:
            q.append(1)
            for j in range(crc_length + 1):
                work[i + j] ^= poly[j]
        else:
            q.append(0)
    check_code = work[-crc_length:]
    return np.array(info_bits + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).tolist()
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length).tolist()
    return expected == bits


def _get_bit(llr_val, is_info):
    if is_info:
        return 0 if llr_val >= 0 else 1
    return 0


def _get_up_loc(bit_matrix):
    n = bit_matrix.shape[0] - 1
    N = bit_matrix.shape[1]
    detect = -1
    for i in range(N):
        if bit_matrix[n][i] != 0 and bit_matrix[n][i] != 1:
            detect = i - 1
            break
    if detect == -1:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _path_metric_update(llr_slice, bit_slice):
    pm = 0.0
    for llr_val, u_bit in zip(llr_slice, bit_slice):
        hard = 0 if llr_val >= 0 else 1
        if u_bit != hard:
            pm += abs(llr_val)
    return pm


def _sc_step_to_split(llr_matrix, bit_matrix, info_positions, frozen_bits, split_pos):
    """SC 译码推进至 split_pos 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    info_set = set(info_positions)
    position = _get_up_loc(bit_matrix) + [n, N]

    while bit_matrix[n][split_pos] not in (0, 1):
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
        elif _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit_val.copy()
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                bit_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = _get_bit(right_llr[0], right_bit_pos in info_set)
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = right_llr_val
        elif not _all_filled(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = left_llr_val
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            bit_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = _get_bit(left_llr[0], left_bit_pos in info_set)
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.sort(np.where(self.frozen_bits == 0)[0])

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        info_pos = self.info_indices.tolist()
        split_pos = info_pos if len(info_pos) > 0 else [N - 1]

        llr_template = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_template[0] = llr_ch
        bit_template = np.full((n + 1, N), np.nan, dtype=np.float64)

        llr_list = [llr_template.copy()]
        bit_list = [bit_template.copy()]
        pm_list = [0.0]
        l_now = 1

        for loc, sp in enumerate(split_pos):
            prev = split_pos[loc - 1] if loc > 0 else -1
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for i in range(l_now):
                llr_m = llr_list[i].copy()
                bit_m = bit_list[i].copy()
                pm_base = pm_list[i]

                llr_m, bit_m = _sc_step_to_split(
                    llr_m, bit_m, info_pos, self.frozen_bits, sp
                )

                llr_slice = llr_m[n][prev + 1 : sp + 1]
                bit_slice = bit_m[n][prev + 1 : sp + 1]
                pm0 = pm_base + _path_metric_update(llr_slice, bit_slice)

                new_llr_list.append(llr_m)
                new_bit_list.append(bit_m)
                new_pm_list.append(pm0)

                if sp in info_pos:
                    bit_flip = bit_m.copy()
                    bit_flip[n][sp] = 1 - bit_flip[n][sp]
                    llr_slice_f = llr_m[n][prev + 1 : sp + 1]
                    bit_slice_f = bit_flip[n][prev + 1 : sp + 1]
                    pm1 = pm_base + _path_metric_update(llr_slice_f, bit_slice_f)
                    new_llr_list.append(llr_m.copy())
                    new_bit_list.append(bit_flip)
                    new_pm_list.append(pm1)

            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]
            l_now = len(pm_list)

        if split_pos[-1] != N - 1:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []
            prev = split_pos[-1]
            for i in range(l_now):
                llr_m = llr_list[i].copy()
                bit_m = bit_list[i].copy()
                pm_base = pm_list[i]
                llr_m, bit_m = _sc_step_to_split(
                    llr_m, bit_m, info_pos, self.frozen_bits, N - 1
                )
                llr_slice = llr_m[n][prev + 1 : N]
                bit_slice = bit_m[n][prev + 1 : N]
                pm = pm_base + _path_metric_update(llr_slice, bit_slice)
                new_llr_list.append(llr_m)
                new_bit_list.append(bit_m)
                new_pm_list.append(pm)
            order = np.argsort(new_pm_list)
            keep = order[: self.list_size]
            llr_list = [new_llr_list[i] for i in keep]
            bit_list = [new_bit_list[i] for i in keep]
            pm_list = [new_pm_list[i] for i in keep]

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_cand = np.nan_to_num(bit_list[idx][n], nan=0).astype(int)
                payload = u_cand[self.info_indices]
                if crc_check(payload, self.crc_length):
                    return u_cand, pm_list[idx]

        best_u = np.nan_to_num(bit_list[order[0]][n], nan=0).astype(int)
        return best_u, best_pm
