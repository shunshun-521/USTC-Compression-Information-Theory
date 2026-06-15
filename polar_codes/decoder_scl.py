"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _permute_llr_for_decode,
    _frozen_mask_to_info,
    _all_num,
    _up,
    _leftdown,
    _rightdown,
    _get_up_bit,
    _get_left_llr,
    _get_right_llr,
    _get_left_bit,
    _get_right_bit,
    f_operation,
    g_operation,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits[:-crc_length], poly, crc_length)
    expected = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.array_equal(bits[-crc_length:], expected)


def _pm_add(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


def _sc_step_to_phi(llr_matrix, bit_matrix, information_pos, frozen_bit, target_phi):
    """将 SC 状态推进到 target_phi 并完成该位判决，返回该位 LLR。"""
    N = llr_matrix.shape[1]
    n = int(np.log2(N))
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0 or np.isnan(bit_matrix[n][target_phi]):
        up_llr = llr_matrix[position[0]][
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
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][
                position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = up_bit_new.copy()
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_pos = position[1] + 1
                if right_pos == target_phi:
                    return float(right_llr[0])
                rb = _get_right_bit(right_llr, information_pos, frozen_bit, right_pos)
                bit_matrix[position[0] + 1][
                    position[1]
                    + 2 ** (position[2] - position[0] - 1) : position[1]
                    + 2 ** (position[2] - position[0])
                ] = rb
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][
                position[1]
                + 2 ** (position[2] - position[0] - 1) : position[1]
                + 2 ** (position[2] - position[0])
            ] = right_llr_new
        elif _all_num(left_llr) == 0:
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][
                position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
            ] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                left_pos = position[1]
                if left_pos == target_phi:
                    return float(left_llr[0])
                lb = _get_left_bit(left_llr, information_pos, frozen_bit, left_pos)
                bit_matrix[position[0] + 1][
                    position[1] : position[1] + 2 ** (position[2] - position[0] - 1)
                ] = lb
            else:
                position = _leftdown(position)

    return 0.0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制矩阵）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.information_pos = _frozen_mask_to_info(self.frozen_bits)
        self.info_set = set(self.information_pos)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_value = 0

    def _new_state(self, llr_ch):
        llr_matrix = np.ones((self.n + 1, self.N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        llr_ch = _permute_llr_for_decode(llr_ch)
        paths = [{"llr": None, "bit": None, "pm": 0.0}]
        paths[0]["llr"], paths[0]["bit"] = self._new_state(llr_ch)

        for phi in range(self.N):
            expanded = []
            for path in paths:
                llr_leaf = _sc_step_to_phi(
                    path["llr"],
                    path["bit"],
                    self.information_pos,
                    self.frozen_value,
                    phi,
                )
                if phi not in self.info_set:
                    bit = self.frozen_value
                    path["bit"][self.n][phi] = bit
                    path["pm"] += _pm_add(llr_leaf, bit)
                    expanded.append(path)
                else:
                    for bit in (0, 1):
                        child = {
                            "llr": path["llr"].copy(),
                            "bit": path["bit"].copy(),
                            "pm": path["pm"] + _pm_add(llr_leaf, bit),
                        }
                        child["bit"][self.n][phi] = bit
                        expanded.append(child)
            expanded.sort(key=lambda p: p["pm"])
            paths = expanded[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            valid = []
            for p in paths:
                u = np.array([0 if p["bit"][self.n][i] == 0 else 1 for i in range(self.N)], dtype=int)
                info_bits = u[self.information_pos]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = paths[0]
        u_hat = np.array([0 if best["bit"][self.n][i] == 0 else 1 for i in range(self.N)], dtype=int)
        for i in range(self.N):
            if self.frozen_bits[i]:
                u_hat[i] = self.frozen_value
        return u_hat, best["pm"]
