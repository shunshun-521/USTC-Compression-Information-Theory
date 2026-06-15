"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _sc_tree_decode,
    _all_decided,
    _up_position,
    _leftdown,
    _rightdown,
    _get_left_llr,
    _get_right_llr,
    _get_up_bit,
)


_CRC_POLY = {
    8: np.array([1, 0, 0, 0, 0, 1, 1, 1], dtype=int),
    16: np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=int),
}


def _crc_remainder(bits, crc_length):
    poly = _CRC_POLY[crc_length]
    msg = list(np.asarray(bits, dtype=int).ravel()) + [0] * crc_length
    n = len(bits)
    for i in range(n):
        if msg[i] == 1:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    return np.array(msg[n : n + crc_length], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length not in _CRC_POLY:
        raise ValueError("crc_length must be 8 or 16")
    crc_bits = _crc_remainder(info_bits, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC_POLY[crc_length]
    msg = list(bits)
    n = len(bits) - crc_length
    for i in range(n):
        if msg[i] == 1:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    return all(x == 0 for x in msg[-crc_length:])


def _path_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(llr)


def _sc_step_to_bit(llr_matrix, bit_matrix, info_positions, frozen_bits, stop_phase):
    """SC 树遍历，直到 bit_matrix[n][stop_phase] 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    position = [0, 0, n, N]

    while np.isnan(bit_matrix[n, stop_phase]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        up_bit = bit_matrix[position[0], position[1] : position[1] + span]
        left_llr = llr_matrix[
            position[0] + 1, position[1] : position[1] + span // 2
        ]
        left_bit = bit_matrix[
            position[0] + 1, position[1] : position[1] + span // 2
        ]
        right_llr = llr_matrix[
            position[0] + 1, position[1] + span // 2 : position[1] + span
        ]
        right_bit = bit_matrix[
            position[0] + 1, position[1] + span // 2 : position[1] + span
        ]

        if _all_decided(up_bit):
            position = _up_position(position)
        elif _all_decided(right_bit):
            merged = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1] : position[1] + span] = merged
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in info_positions:
                    right_bit_val = 0 if right_llr[0] >= 0 else 1
                else:
                    right_bit_val = 0
                bit_matrix[
                    position[0] + 1,
                    position[1] + span // 2 : position[1] + span,
                ] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_decided(left_bit):
            new_right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[
                position[0] + 1,
                position[1] + span // 2 : position[1] + span,
            ] = new_right_llr
        elif not _all_decided(left_llr):
            new_left_llr = _get_left_llr(up_llr)
            llr_matrix[
                position[0] + 1, position[1] : position[1] + span // 2
            ] = new_left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in info_positions:
                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    left_bit_val = 0
                bit_matrix[
                    position[0] + 1, position[1] : position[1] + span // 2
                ] = left_bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_positions = set(np.where(~self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        n, N = self.n, self.N

        if self.list_size == 1:
            u_hat = _sc_tree_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr0 = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit0 = np.full((n + 1, N), np.nan)
        llr0[0] = llr_ch

        paths = [(0.0, llr0, bit0)]

        for phi in range(N):
            new_paths = []
            for pm, llr_m, bit_m in paths:
                llr_m, bit_m = _sc_step_to_bit(
                    llr_m, bit_m, self.info_positions, self.frozen_bits, phi
                )
                llr_leaf = llr_m[n, phi]
                if np.isnan(llr_leaf):
                    llr_leaf = 0.0

                if phi in self.info_positions:
                    for bit in (0, 1):
                        llr_c = copy.deepcopy(llr_m)
                        bit_c = copy.deepcopy(bit_m)
                        bit_c[n, phi] = bit
                        new_pm = pm + _path_penalty(llr_leaf, bit)
                        new_paths.append((new_pm, llr_c, bit_c))
                else:
                    bit_m[n, phi] = 0
                    pm += _path_penalty(llr_leaf, 0)
                    new_paths.append((pm, llr_m, bit_m))

            new_paths.sort(key=lambda x: x[0])
            paths = new_paths[: self.list_size]

        for phi in range(N):
            if any(np.isnan(paths[0][2][n])):
                for i, (pm, llr_m, bit_m) in enumerate(paths):
                    llr_m, bit_m = _sc_step_to_bit(
                        llr_m, bit_m, self.info_positions, self.frozen_bits, phi
                    )
                    paths[i] = (pm, llr_m, bit_m)

        paths.sort(key=lambda x: x[0])
        info_idx = sorted(self.info_positions)

        if self.crc_length > 0:
            for pm, _, bit_m in paths:
                u_hat = bit_m[n].astype(int)
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    return u_hat, pm

        best_pm, _, best_bits = paths[0]
        return best_bits[n].astype(int), best_pm
