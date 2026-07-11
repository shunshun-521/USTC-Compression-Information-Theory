"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import (
    sc_decode,
    sc_stepping_decoder,
    _pm_update,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    top_bit = 1 << (crc_length - 1)
    reg = 0

    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    for _ in range(crc_length):
        if reg & top_bit:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(
        bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    )


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_pos = np.where(~self.frozen_bits)[0].tolist()
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_matrices(self, llr_ch):
        llr_matrix = np.ones((self.n + 1, self.N), dtype=np.float64)
        llr_matrix[:] = np.nan
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = np.asarray(llr_ch, dtype=np.float64)
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        info_pos = self.info_pos
        frozen_bit = 0
        llr_list = []
        bit_list = []
        llr_m, bit_m = self._new_matrices(llr_ch)
        llr_list.append(llr_m)
        bit_list.append(bit_m)
        pm_list = [0.0]

        split_pos = info_pos if self.crc_length == 0 else info_pos
        if not split_pos:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        for split_idx, sp in enumerate(split_pos):
            new_llr, new_bit, new_pm = [], [], []
            for llr_m, bit_m, pm in zip(llr_list, bit_list, pm_list):
                llr_t = llr_m.copy()
                bit_t = bit_m.copy()
                llr_t, bit_t = sc_stepping_decoder(
                    llr_t, bit_t, info_pos, frozen_bit, sp
                )
                new_llr.append(llr_t)
                new_bit.append(bit_t)
                new_pm.append(pm)

                if not self.frozen_bits[sp]:
                    bit_wrong = bit_t.copy()
                    bit_wrong[self.n][sp] = 1 - bit_wrong[self.n][sp]
                    prev = split_pos[split_idx - 1] + 1 if split_idx > 0 else 0
                    wrong_pm = _pm_update(
                        llr_t[self.n][prev : sp + 1],
                        bit_wrong[self.n][prev : sp + 1],
                    )
                    new_llr.append(llr_t.copy())
                    new_bit.append(bit_wrong)
                    new_pm.append(pm + wrong_pm)

            order = np.argsort(new_pm)
            keep = order[: self.list_size]
            llr_list = [new_llr[i] for i in keep]
            bit_list = [new_bit[i] for i in keep]
            pm_list = [new_pm[i] for i in keep]

        if split_pos[-1] != self.N - 1:
            for i in range(len(llr_list)):
                llr_list[i], bit_list[i] = sc_stepping_decoder(
                    llr_list[i], bit_list[i], info_pos, frozen_bit, self.N - 1
                )
                prev = split_pos[-1] + 1
                pm_list[i] += _pm_update(
                    llr_list[i][self.n][prev : self.N],
                    bit_list[i][self.n][prev : self.N],
                )

        order = np.argsort(pm_list)
        best_u = None
        best_pm = pm_list[order[0]]

        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][self.n].astype(int)
                if crc_check(u_cand, self.crc_length):
                    best_u = u_cand
                    best_pm = pm_list[idx]
                    break

        if best_u is None:
            best_u = bit_list[order[0]][self.n].astype(int)
            best_pm = pm_list[order[0]]

        best_u[self.frozen_bits] = 0
        return best_u, best_pm
