"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation, g_operation, _all_decided, _up, _leftdown, _rightdown,
    _get_left_llr, _get_right_llr, _get_up_bit, sc_decode_recursive,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_polynomial(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError(f'Unsupported CRC length: {crc_length}')


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_polynomial(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


def _sc_step_to_bit(llr_matrix, bit_matrix, frozen_bits, target_bit, info_positions):
    """将 SC 树推进到 target_bit 判决完成。"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]

    while np.isnan(bit_matrix[n, target_bit]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
            continue
        if _all_decided(right_bit):
            bit_matrix[position[0]][position[1]:position[1] + span] = _get_up_bit(left_bit, right_bit)
            continue
        if _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_pos = position[1] + half
                if frozen_bits[right_pos]:
                    val = 0
                else:
                    val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = val
            else:
                position = _rightdown(position)
            continue
        if _all_decided(left_bit):
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = _get_right_llr(left_bit, up_llr)
            continue
        if not _all_decided(left_llr):
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = _get_left_llr(up_llr)
            continue
        if position[0] == position[2] - 1:
            left_pos = position[1]
            if frozen_bits[left_pos]:
                val = 0
            else:
                val = 0 if left_llr[0] >= 0 else 1
            bit_matrix[position[0] + 1][position[1]:position[1] + half] = val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        N = self.N
        n = self.n
        brp = bit_reversal_permutation(N)
        llr = np.asarray(llr_ch, dtype=np.float64)[brp]

        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr

        paths = [{'llr': llr_matrix.copy(), 'bit': bit_matrix.copy(), 'pm': 0.0}]

        for phi in self.info_positions:
            new_paths = []
            for path in paths:
                llr_m, bit_m = _sc_step_to_bit(
                    path['llr'].copy(), path['bit'].copy(),
                    self.frozen_bits, phi, self.info_positions
                )
                llr_val = llr_m[n, phi]
                if self.frozen_bits[phi]:
                    pm = _path_metric_update(path['pm'], llr_val, 0)
                    bit_m[n, phi] = 0
                    new_paths.append({'llr': llr_m, 'bit': bit_m, 'pm': pm})
                else:
                    for u in (0, 1):
                        llr_c = path['llr'].copy()
                        bit_c = path['bit'].copy()
                        llr_c, bit_c = _sc_step_to_bit(
                            llr_c, bit_c, self.frozen_bits, phi, self.info_positions
                        )
                        pm = _path_metric_update(path['pm'], llr_c[n, phi], u)
                        bit_c[n, phi] = u
                        new_paths.append({'llr': llr_c, 'bit': bit_c, 'pm': pm})

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:self.list_size]

        if paths[0]['bit'][n, self.info_positions[-1]] is np.nan or \
           np.isnan(paths[0]['bit'][n, self.info_positions[-1]]):
            for path in paths:
                _, bit_m = _sc_step_to_bit(
                    path['llr'].copy(), path['bit'].copy(),
                    self.frozen_bits, self.info_positions[-1], self.info_positions
                )
                path['bit'] = bit_m

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = np.nan_to_num(path['bit'][n], nan=0).astype(int)
                if crc_check(u_hat, self.crc_length):
                    valid.append(path)
            best = min(valid, key=lambda p: p['pm']) if valid else paths[0]
        else:
            best = paths[0]

        u_hat = np.nan_to_num(best['bit'][n], nan=0).astype(int)
        return u_hat, best['pm']
