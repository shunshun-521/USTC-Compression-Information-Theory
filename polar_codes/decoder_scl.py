"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import _sc_decoder_core, sc_decode


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def _crc_register(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_register(padded, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _pm_penalty(llr_val, bit):
    hard = 0 if llr_val >= 0 else 1
    return abs(llr_val) if hard != bit else 0.0


def _scl_decoder_core(y_llr, information_pos, frozen_bit, list_size):
    """在 SC 因子图译码过程中维护最多 list_size 条路径。"""
    from decoder_sc import (
        _all_filled,
        _get_left_bit,
        _get_left_llr,
        _get_right_bit,
        _get_right_llr,
        _get_up_bit,
        _leftdown,
        _rightdown,
        _up,
    )

    N = len(y_llr)
    n = int(math.log2(N))
    info_set = set(int(i) for i in information_pos)

    def new_state():
        llr_m = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_m = np.full((n + 1, N), np.nan)
        llr_m[0] = y_llr
        return {
            "llr": llr_m,
            "bit": bit_m,
            "pos": [0, 0, n, N],
            "pm": 0.0,
        }

    paths = [new_state()]

    def decide_bit(state, bit_pos, llr_val, forced_bit=None):
        if forced_bit is not None:
            return [(forced_bit, 0.0)]
        if bit_pos in info_set:
            hard = 0 if llr_val >= 0 else 1
            opts = [(hard, 0.0)]
            alt = 1 - hard
            opts.append((alt, _pm_penalty(llr_val, alt)))
            return opts
        return [(frozen_bit, 0.0)]

    def run_step(state):
        position = state["pos"]
        llr_matrix = state["llr"]
        bit_matrix = state["bit"]

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

        forks = []

        if _all_filled(up_bit):
            state["pos"] = _up(position)
            return [state]
        if _all_filled(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            sl = slice(position[1], position[1] + 2 ** (position[2] - position[0]))
            bit_matrix[position[0]][sl] = up_bit_val
            return [state]
        if _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                for bit_val, penalty in decide_bit(state, right_bit_pos, right_llr[0]):
                    st = copy.deepcopy(state)
                    rs = slice(
                        position[1] + 2 ** (position[2] - position[0] - 1),
                        position[1] + 2 ** (position[2] - position[0]),
                    )
                    st["bit"][position[0] + 1][rs] = bit_val
                    st["pm"] += penalty
                    forks.append(st)
                return forks
            state["pos"] = _rightdown(position)
            return [state]
        if _all_filled(left_bit):
            right_llr_new = _get_right_llr(left_bit, up_llr)
            rs = slice(
                position[1] + 2 ** (position[2] - position[0] - 1),
                position[1] + 2 ** (position[2] - position[0]),
            )
            llr_matrix[position[0] + 1][rs] = right_llr_new
            return [state]
        if not _all_filled(left_llr):
            left_llr_new = _get_left_llr(up_llr)
            ls = slice(position[1], position[1] + 2 ** (position[2] - position[0] - 1))
            llr_matrix[position[0] + 1][ls] = left_llr_new
            return [state]
        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            for bit_val, penalty in decide_bit(state, left_bit_pos, left_llr[0]):
                st = copy.deepcopy(state)
                ls = slice(position[1], position[1] + 2 ** (position[2] - position[0] - 1))
                st["bit"][position[0] + 1][ls] = bit_val
                st["pm"] += penalty
                forks.append(st)
            return forks
        state["pos"] = _leftdown(position)
        return [state]

    def all_done(st):
        return _all_filled(st["bit"][n])

    while not all(all_done(state) for state in paths):
        new_paths = []
        for state in paths:
            if all_done(state):
                new_paths.append(state)
                continue
            new_paths.extend(run_step(state))
        new_paths.sort(key=lambda s: s["pm"])
        paths = new_paths[:list_size]

    best = min(paths, key=lambda s: s["pm"])
    return best["bit"][n].astype(int), best["pm"]


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        u_hat, pm = _scl_decoder_core(
            llr_ch, self.information_pos, 0, self.list_size
        )

        if self.crc_length > 0:
            info_idx = self.information_pos
            if not crc_check(u_hat[info_idx], self.crc_length):
                u_alt, pm_alt = _scl_decoder_core(
                    llr_ch, self.information_pos, 0, self.list_size * 2
                )
                if crc_check(u_alt[info_idx], self.crc_length):
                    return u_alt, pm_alt

        return u_hat, pm
