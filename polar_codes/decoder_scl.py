"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _reorder_channel_llr,
    _frozen_bits_to_info_set,
    _all_computed,
    _leftdown,
    _rightdown,
    _up,
    _get_up_bit,
    _get_right_llr,
    _get_left_llr,
    _get_left_bit,
    _get_right_bit,
)


def _build_crc_poly(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError("crc_length must be 8 or 16")
    poly = [0] * (crc_length + 1)
    for i in loc:
        poly[i] = 1
    return poly[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = [int(b) for b in np.asarray(info_bits, dtype=int)]
    poly = _build_crc_poly(crc_length)
    work = info_bits + [0] * crc_length
    for i in range(len(info_bits)):
        if work[i] == 1:
            for j in range(crc_length + 1):
                work[i + j] ^= poly[j]
    check = work[-crc_length:]
    return np.array(info_bits + check, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = [int(b) for b in np.asarray(bits, dtype=int)]
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return list(expected) == bits


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


def _sc_step_to(llr_matrix, bit_matrix, info_set, target_phi):
    """将 SC 树推进到完成 target_phi 比特判决"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]

    while np.isnan(bit_matrix[n, target_phi]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_val[0]
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + half
                bit_matrix[position[0] + 1, position[1] + half] = _get_right_bit(
                    right_llr[0], info_set, right_bit_pos
                )
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr_val = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr_val
        elif not _all_computed(left_llr):
            left_llr_val = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                bit_matrix[position[0] + 1, position[1]] = _get_left_bit(
                    left_llr[0], info_set, left_bit_pos
                )
            else:
                position = _leftdown(position)


def _init_matrices(llr_ch, N, n):
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_set = _frozen_bits_to_info_set(self.frozen_bits)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = _reorder_channel_llr(llr_ch, self.N)
        N, n = self.N, self.n

        llr0, bit0 = _init_matrices(llr_ch, N, n)
        paths = [{"pm": 0.0, "llr": llr0, "bit": bit0, "u": np.zeros(N, dtype=int)}]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr_m = path["llr"]
                bit_m = path["bit"]
                _sc_step_to(llr_m, bit_m, self.info_set, phi)
                llr_phi = llr_m[n, phi]

                if self.frozen_bits[phi]:
                    pm = path["pm"] + _pm_penalty(llr_phi, 0)
                    new_bit = bit_m.copy()
                    new_bit[n, phi] = 0
                    new_u = path["u"].copy()
                    new_u[phi] = 0
                    new_paths.append(
                        {"pm": pm, "llr": llr_m, "bit": new_bit, "u": new_u}
                    )
                else:
                    for u in (0, 1):
                        pm = path["pm"] + _pm_penalty(llr_phi, u)
                        new_bit = bit_m.copy()
                        new_bit[n, phi] = u
                        new_u = path["u"].copy()
                        new_u[phi] = u
                        new_paths.append(
                            {
                                "pm": pm,
                                "llr": llr_m.copy(),
                                "bit": new_bit,
                                "u": new_u,
                            }
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            best = min(valid, key=lambda p: p["pm"]) if valid else min(
                paths, key=lambda p: p["pm"]
            )
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"], best["pm"]
