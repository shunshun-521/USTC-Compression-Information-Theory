"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from sc_core import (
    preprocess_llr_for_polar_encode,
    sc_tree_decode,
    sc_step_to_position,
    get_pm_update,
)
from decoder_sc import _frozen_to_info


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(info_bits, crc_length):
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        if crc_length == 8:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        else:
            reg ^= int(b) << 15
            for _ in range(16):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ poly) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
    if crc_length == 8:
        return np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    return np.concatenate([info_bits, _crc_bits(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（路径维护 LLR/比特矩阵，Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        assert 2 ** self.n == N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = _frozen_to_info(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length

    def _init_matrices(self, llr_ch):
        llr_matrix = np.full((self.n + 1, self.N), np.nan)
        bit_matrix = np.full((self.n + 1, self.N), np.nan)
        llr_matrix[0] = llr_ch
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        llr_ch = preprocess_llr_for_polar_encode(llr_ch)
        info_pos = list(self.info_indices)
        frozen_val = 0

        llr_list = [self._init_matrices(llr_ch)[0]]
        bit_list = [self._init_matrices(llr_ch)[1]]
        pm_list = [0.0]

        split_loc = 0
        n_splits = len(info_pos)

        while split_loc < n_splits:
            split_pos = info_pos[split_loc]
            new_llr, new_bit, new_pm = [], [], []

            for idx in range(len(llr_list)):
                llr_m = llr_list[idx].copy()
                bit_m = bit_list[idx].copy()
                pm = pm_list[idx]

                llr_m, bit_m = sc_step_to_position(
                    llr_m, bit_m, info_pos, frozen_val, split_pos
                )

                prev_pos = info_pos[split_loc - 1] if split_loc > 0 else -1
                seg_slice = slice(prev_pos + 1, split_pos + 1)
                seg_llr = llr_m[self.n][seg_slice]
                seg_bit = bit_m[self.n][seg_slice]

                pm0 = pm + get_pm_update(seg_llr, seg_bit)
                new_llr.append(llr_m)
                new_bit.append(bit_m)
                new_pm.append(pm0)

                bit_wrong = bit_m.copy()
                bit_wrong[self.n][split_pos] = 1 - bit_wrong[self.n][split_pos]
                seg_bit_w = bit_wrong[self.n][seg_slice]
                pm1 = pm + get_pm_update(seg_llr, seg_bit_w)
                new_llr.append(llr_m.copy())
                new_bit.append(bit_wrong)
                new_pm.append(pm1)

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]
            split_loc += 1

        if info_pos:
            last_pos = info_pos[-1]
            for idx in range(len(llr_list)):
                llr_m, bit_m = sc_step_to_position(
                    llr_list[idx], bit_list[idx], info_pos, frozen_val, self.N - 1
                )
                prev_pos = info_pos[-2] if len(info_pos) > 1 else -1
                seg_slice = slice(prev_pos + 1, self.N)
                pm_list[idx] += get_pm_update(
                    llr_m[self.n][seg_slice], bit_m[self.n][seg_slice]
                )
                llr_list[idx] = llr_m
                bit_list[idx] = bit_m

        order = np.argsort(pm_list)
        candidates = [(bit_list[i][self.n].astype(int), pm_list[i]) for i in order]

        if self.crc_length > 0:
            for u_hat, pm in candidates:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat.copy(), pm

        best_u, best_pm = candidates[0]
        return best_u.copy(), best_pm
