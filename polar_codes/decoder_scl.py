"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _all_ready,
    _get_left_bit,
    _get_left_llr,
    _get_right_bit,
    _get_right_llr,
    _get_up_bit,
    _get_up_loc,
    _leftdown,
    _preprocess_llr,
    _rightdown,
    _sc_decode_core,
    _up_position,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]
_CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1]


def _poly_remainder(msg_bits, gen_poly):
    msg = [int(b) for b in msg_bits]
    n = len(gen_poly)
    for i in range(len(msg) - n + 1):
        if msg[i] == 1:
            for j in range(n):
                msg[i + j] ^= gen_poly[j]
    return np.array(msg[-(n - 1):], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        gen = _CRC8_POLY
    elif crc_length == 16:
        gen = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    remainder = _poly_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), gen
    )
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        gen = _CRC8_POLY
    elif crc_length == 16:
        gen = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.all(_poly_remainder(bits, gen) == 0)


def _sc_step_to_bit(llr_matrix, bit_matrix, information_pos, frozen_bit, split_pos):
    """推进 SC 状态直到 split_pos 完成判决。"""
    N = bit_matrix.shape[1]
    n = int(np.log2(N))
    information_pos = set(information_pos)
    position = _get_up_loc(bit_matrix)
    max_steps = 4 * N * n
    steps = 0

    while bit_matrix[n, split_pos] != 0 and bit_matrix[n, split_pos] != 1:
        steps += 1
        if steps > max_steps or position[0] < 0 or position[0] >= n:
            break

        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start:start + span]
        up_bit = bit_matrix[position[0], start:start + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, start:start + half]
        left_bit = bit_matrix[position[0] + 1, start:start + half]
        right_llr = llr_matrix[position[0] + 1, start + half:start + span]
        right_bit = bit_matrix[position[0] + 1, start + half:start + span]

        if _all_ready(up_bit):
            position = _up_position(position)
        elif _all_ready(right_bit):
            bit_matrix[position[0], start:start + span] = _get_up_bit(left_bit, right_bit)
        elif _all_ready(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = start + 1
                bit_val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1, start + half] = bit_val
            else:
                position = _rightdown(position)
        elif _all_ready(left_bit):
            llr_matrix[position[0] + 1, start + half:start + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_ready(left_llr):
            llr_matrix[position[0] + 1, start:start + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = start
                bit_val = _get_left_bit(
                    left_llr[0], information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1, start] = bit_val
            else:
                position = _leftdown(position)

    return llr_matrix, bit_matrix


def _path_metric(llr_slice, bit_slice):
    pm = 0.0
    for llr, bit in zip(llr_slice, bit_slice):
        if (0 if llr >= 0 else 1) != int(bit):
            pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.information_pos = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_state(self, llr):
        llr_matrix = np.ones((self.n + 1, self.N), dtype=np.float64)
        llr_matrix[llr_matrix == 1] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0, :] = llr
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = _preprocess_llr(llr_ch)

        if self.list_size == 1:
            u_hat = _sc_decode_core(llr, self.information_pos, frozen_bit=0)
            return u_hat, 0.0

        llr_list, bit_list, pm_list = [], [], []
        llr_m, bit_m = self._init_state(llr)
        llr_list.append(llr_m)
        bit_list.append(bit_m)
        pm_list.append(0.0)

        prev_pos = -1
        for split_pos in self.information_pos:
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_temp = llr_m.copy()
                bit_temp = bit_m.copy()
                llr_temp, bit_temp = _sc_step_to_bit(
                    llr_temp, bit_temp, self.information_pos, 0, split_pos
                )
                llr_slice = llr_temp[self.n, prev_pos + 1:split_pos + 1]
                bit_slice = bit_temp[self.n, prev_pos + 1:split_pos + 1]
                pm_add = _path_metric(llr_slice, bit_slice)

                for bit in (0, 1):
                    llr_copy = llr_temp.copy()
                    bit_copy = bit_temp.copy()
                    bit_copy[self.n, split_pos] = bit
                    llr_val = llr_copy[self.n, split_pos]
                    penalty = 0.0 if (0 if llr_val >= 0 else 1) == bit else abs(llr_val)
                    new_llr.append(llr_copy)
                    new_bit.append(bit_copy)
                    new_pm.append(pm + pm_add + penalty)

            order = np.argsort(new_pm)[: self.list_size]
            llr_list = [new_llr[i] for i in order]
            bit_list = [new_bit[i] for i in order]
            pm_list = [new_pm[i] for i in order]
            prev_pos = split_pos

        final_bit, final_pm = [], []
        for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
            llr_temp = llr_m.copy()
            bit_temp = bit_m.copy()
            llr_temp, bit_temp = _sc_step_to_bit(
                llr_temp, bit_temp, self.information_pos, 0, self.N - 1
            )
            llr_slice = llr_temp[self.n, prev_pos + 1:self.N]
            bit_slice = bit_temp[self.n, prev_pos + 1:self.N]
            final_bit.append(bit_temp)
            final_pm.append(pm + _path_metric(llr_slice, bit_slice))

        order = np.argsort(final_pm)
        if self.crc_length > 0:
            for idx in order:
                u_hat = final_bit[idx][self.n, :].astype(int)
                if crc_check(u_hat[self.information_pos], self.crc_length):
                    return u_hat, final_pm[idx]

        best_idx = order[0]
        return final_bit[best_idx][self.n, :].astype(int), final_pm[best_idx]
