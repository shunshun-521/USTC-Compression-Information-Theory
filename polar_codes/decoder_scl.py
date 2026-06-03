"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
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
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length <= 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _pm_update(llr_slice, bit_slice):
    """路径度量：与 LLR 符号不一致时加 |LLR|"""
    pm = 0.0
    for llr, b in zip(llr_slice, bit_slice):
        hard = 0 if llr >= 0 else 1
        if hard != int(b):
            pm += abs(llr)
    return pm


def _sc_step_to_split(llr_matrix, bit_matrix, info_list, frozen_val, split_pos):
    """SC 单步译码至分裂点 split_pos（含）"""
    N = llr_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1 : p1 + half]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half : p1 + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        else:
            if _all_num(right_bit) == 1:
                up_bit_row = _get_up_bit(left_bit, right_bit)
                bit_matrix[p0][p1 : p1 + span] = up_bit_row.copy()
            else:
                if _all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_val = _get_right_bit(right_llr, info_list, frozen_val, p1 + half)
                        bit_matrix[p0 + 1][p1 + half : p1 + span] = right_bit_val
                        if p1 + half == split_pos:
                            return llr_matrix, bit_matrix
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit) == 1:
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[p0 + 1][p1 + half : p1 + span] = right_llr_new
                    else:
                        if _all_num(left_llr) == 0:
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[p0 + 1][p1 : p1 + half] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_val = _get_left_bit(left_llr, info_list, frozen_val, p1)
                                bit_matrix[p0 + 1][p1 : p1 + half] = left_bit_val
                                if p1 == split_pos:
                                    return llr_matrix, bit_matrix
                            else:
                                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _sc_finish(llr_matrix, bit_matrix, info_list, frozen_val):
    """完成剩余 SC 译码"""
    N = llr_matrix.shape[1]
    n = int(math.log2(N))
    position = [0, 0, n, N]
    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        half = span // 2
        left_llr = llr_matrix[p0 + 1][p1 : p1 + half]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half : p1 + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        else:
            if _all_num(right_bit) == 1:
                up_bit_row = _get_up_bit(left_bit, right_bit)
                bit_matrix[p0][p1 : p1 + span] = up_bit_row.copy()
            else:
                if _all_num(right_llr) == 1:
                    if position[0] == position[2] - 1:
                        right_bit_val = _get_right_bit(right_llr, info_list, frozen_val, p1 + half)
                        bit_matrix[p0 + 1][p1 + half : p1 + span] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_num(left_bit) == 1:
                        right_llr_new = _get_right_llr(left_bit, up_llr)
                        llr_matrix[p0 + 1][p1 + half : p1 + span] = right_llr_new
                    else:
                        if _all_num(left_llr) == 0:
                            left_llr_new = _get_left_llr(up_llr)
                            llr_matrix[p0 + 1][p1 : p1 + half] = left_llr_new
                        else:
                            if position[0] == position[2] - 1:
                                left_bit_val = _get_left_bit(left_llr, info_list, frozen_val, p1)
                                bit_matrix[p0 + 1][p1 : p1 + half] = left_bit_val
                            else:
                                position = _leftdown(position)
    return bit_matrix[n].astype(int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy：路径分裂时复制矩阵）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_list = list(self.info_indices.astype(int))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_val = 0

    def _init_matrices(self, llr_ch):
        llr_matrix = np.full((self.n + 1, self.N), np.nan, dtype=np.float64)
        bit_matrix = np.full((self.n + 1, self.N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        info_only = [i for i in self.info_list]

        llr_list = [self._init_matrices(llr_ch)[0]]
        bit_list = [self._init_matrices(llr_ch)[1]]
        pm_list = [0.0]

        for split_pos in info_only:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_c = llr_m.copy()
                bit_c = bit_m.copy()
                llr_c, bit_c = _sc_step_to_split(llr_c, bit_c, self.info_list, self.frozen_val, split_pos)

                llr_at = llr_c[self.n][split_pos]
                for u_try in (0, 1):
                    llr_mc = llr_c.copy()
                    bit_mc = bit_c.copy()
                    bit_mc[self.n][split_pos] = u_try
                    penalty = 0.0 if (llr_at >= 0) == (u_try == 0) else abs(llr_at)
                    new_llr.append(llr_mc)
                    new_bit.append(bit_mc)
                    new_pm.append(pm + penalty)

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]

        best_u, best_pm = None, float("inf")
        crc_pass = []

        for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
            u_hat = _sc_finish(llr_m, bit_m, self.info_list, self.frozen_val)
            if self.crc_length > 0:
                payload_idx = self.info_indices[: len(self.info_indices) - self.crc_length]
                if crc_check(u_hat[payload_idx], self.crc_length) or crc_check(
                    u_hat[self.info_indices], self.crc_length
                ):
                    crc_pass.append((pm, u_hat))
            if pm < best_pm:
                best_pm = pm
                best_u = u_hat

        if self.crc_length > 0 and crc_pass:
            crc_pass.sort(key=lambda x: x[0])
            return crc_pass[0][1], crc_pass[0][0]

        return best_u, best_pm
