"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    _unpermute_llr, _frozen_to_info_set, _all_num,
    _leftdown, _rightdown, _up, _get_up_bit,
    _get_left_llr, _get_right_llr, _get_left_bit, _get_right_bit,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg <<= 1
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_penalty(llr, u_bit):
    u_from_llr = 0 if llr >= 0 else 1
    return 0.0 if u_bit == u_from_llr else abs(llr)


def _get_up_loc(bit_matrix):
    """定位树遍历续传位置"""
    N = bit_matrix.shape[1]
    n = bit_matrix.shape[0] - 1
    detect_array = bit_matrix[n]
    if np.all(np.isnan(detect_array)):
        return [0, 0]
    detect = 0
    for i in range(N):
        if detect_array[i] != 0 and detect_array[i] != 1:
            detect = i - 1
            break
    else:
        detect = N - 1
    if detect < 0:
        return [0, 0]
    if detect % 2 == 0:
        return [n - 1, detect]
    return [n - 1, detect - 1]


def _sc_step_to_bit(llr_matrix, bit_matrix, info_set, frozen_bit, target_bit):
    """将 SC 树推进到 target_bit 并完成该位判决"""
    N = bit_matrix.shape[1]
    n = bit_matrix.shape[0] - 1
    loc = _get_up_loc(bit_matrix)
    position = [loc[0], loc[1], n, N]

    while bit_matrix[n, target_bit] != 0 and bit_matrix[n, target_bit] != 1:
        span = 2 ** (position[2] - position[0])
        sl = position[1]
        sr = sl + span
        up_llr = llr_matrix[position[0], sl:sr]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, sl:sl + half]
        left_bit = bit_matrix[position[0] + 1, sl:sl + half]
        right_llr = llr_matrix[position[0] + 1, sl + half:sr]
        right_bit = bit_matrix[position[0] + 1, sl + half:sr]

        if _all_num(bit_matrix[position[0], sl:sr]) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            bit_matrix[position[0], sl:sr] = _get_up_bit(left_bit, right_bit)
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_pos = position[1] + 1
                val = _get_right_bit(right_llr, info_set, frozen_bit, right_pos)
                bit_matrix[position[0] + 1, sl + half:sr] = val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            llr_matrix[position[0] + 1, sl + half:sr] = _get_right_llr(left_bit, up_llr)
        elif _all_num(left_llr) == 0:
            llr_matrix[position[0] + 1, sl:sl + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_pos = position[1]
                val = _get_left_bit(left_llr, info_set, frozen_bit, left_pos)
                bit_matrix[position[0] + 1, sl:sl + half] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _init_matrices(y_llr):
    N = len(y_llr)
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_set = _frozen_to_info_set(frozen_bits)

    def decode(self, llr_ch):
        y_llr = _unpermute_llr(llr_ch)
        N = self.N
        n = self.n

        paths = [{
            'llr': _init_matrices(y_llr)[0],
            'bit': _init_matrices(y_llr)[1],
            'pm': 0.0,
        }]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr_m = path['llr'].copy()
                bit_m = path['bit'].copy()
                llr_m, bit_m = _sc_step_to_bit(
                    llr_m, bit_m, self.info_set, 0, phi
                )
                llr_phi = llr_m[n, phi]

                if self.frozen_bits[phi]:
                    pm = path['pm'] + _pm_penalty(llr_phi, 0)
                    bit_m[n, phi] = 0
                    new_paths.append({'llr': llr_m, 'bit': bit_m, 'pm': pm})
                else:
                    for u_bit in (0, 1):
                        bm = bit_m.copy()
                        bm[n, phi] = u_bit
                        pm = path['pm'] + _pm_penalty(llr_phi, u_bit)
                        new_paths.append({'llr': llr_m.copy(), 'bit': bm, 'pm': pm})

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths
                     if crc_check(p['bit'][n].astype(int), self.crc_length)]
            best = min(valid, key=lambda p: p['pm']) if valid else paths[0]
        else:
            best = paths[0]

        return best['bit'][n].astype(int), best['pm']
