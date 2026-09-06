"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import sc_decode


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_len + 1)) - 1)
        if reg & (1 << crc_len):
            reg ^= poly
    return reg & ((1 << crc_len) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


def _init_matrices(llr_ch, n, N):
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr_ch
    return llr_matrix, bit_matrix


def _sc_step(llr_matrix, bit_matrix, frozen_bits, stop_idx):
    """SC 译码直到 bit_matrix[n][stop_idx] 被判决"""
    from decoder_sc import (
        _all_decided,
        _get_left_llr,
        _get_right_llr,
        _get_up_bit,
        _leftdown,
        _rightdown,
        _up_position,
    )

    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]

    while np.isnan(bit_matrix[n, stop_idx]):
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
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0]),
        ]
        right_bit = bit_matrix[
            position[0] + 1,
            position[1] + 2 ** (position[2] - position[0] - 1) : position[1] + 2 ** (position[2] - position[0]),
        ]

        if _all_decided(up_bit):
            position = _up_position(position)
        elif _all_decided(right_bit):
            merged = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[
                position[0], position[1] : position[1] + 2 ** (position[2] - position[0])
            ] = merged
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                idx = position[1] + 2 ** (position[2] - position[0] - 1)
                if frozen_bits[idx]:
                    val = 0
                else:
                    val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1, idx] = val
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            new_right = _get_right_llr(left_bit.astype(int), up_llr)
            sl = slice(
                position[1] + 2 ** (position[2] - position[0] - 1),
                position[1] + 2 ** (position[2] - position[0]),
            )
            llr_matrix[position[0] + 1, sl] = new_right
        elif np.any(np.isnan(left_llr)):
            new_left = _get_left_llr(up_llr)
            sl = slice(position[1], position[1] + 2 ** (position[2] - position[0] - 1))
            llr_matrix[position[0] + 1, sl] = new_left
        else:
            if position[0] == position[2] - 1:
                idx = position[1]
                if frozen_bits[idx]:
                    val = 0
                else:
                    val = 0 if left_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1, idx] = val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric(llr_matrix, bit_matrix, bit_idx, bit_val, frozen):
    llr_val = llr_matrix[0, bit_idx]
    if frozen:
        return 0.0 if llr_val >= 0 else abs(llr_val)
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if bit_val == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = []
        llr_m, bit_m = _init_matrices(llr_ch, n, N)
        paths.append((llr_m, bit_m, 0.0))

        for bit_idx in range(N):
            new_paths = []
            for llr_m, bit_m, pm in paths:
                llr_m, bit_m = _sc_step(llr_m.copy(), bit_m.copy(), self.frozen_bits, bit_idx)
                if self.frozen_bits[bit_idx]:
                    bit_m[n, bit_idx] = 0
                    new_paths.append((llr_m, bit_m, pm + _path_metric(llr_m, bit_m, bit_idx, 0, True)))
                else:
                    for bit_val in (0, 1):
                        bm = bit_m.copy()
                        bm[n, bit_idx] = bit_val
                        penalty = _path_metric(llr_m, bit_m, bit_idx, bit_val, False)
                        new_paths.append((llr_m.copy(), bm, pm + penalty))

            new_paths.sort(key=lambda x: x[2])
            paths = new_paths[: self.list_size]

        candidates = [(pm, bit_m[n].astype(int)) for _, bit_m, pm in paths]

        if self.crc_length > 0:
            for pm, u_hat in sorted(candidates, key=lambda x: x[0]):
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    return u_hat, pm

        best_pm, best_u = min(candidates, key=lambda x: x[0])
        return best_u, best_pm
