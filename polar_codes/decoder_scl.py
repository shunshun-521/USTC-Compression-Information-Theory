"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import sc_decode, sc_stepping_decoder, _pm_update_segment

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = CRC8_POLY
        mask = 0xFF
    elif crc_length == 16:
        poly = CRC16_POLY
        mask = 0xFFFF
    else:
        raise ValueError("crc_length must be 8 or 16")

    rem = 0
    for bit in info_bits:
        rem ^= int(bit) << (crc_length - 1)
        if rem & (1 << (crc_length - 1)):
            rem = ((rem << 1) ^ poly) & mask
        else:
            rem = (rem << 1) & mask

    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class SCLDecoder:
    """SCL 译码器（基于 phased SC 步进）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = np.sort(np.where(self.frozen_bits == 0)[0])

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        n = self.n
        N = self.N
        llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
        llr_matrix[0] = llr_ch

        llr_list = [llr_matrix.copy()]
        bit_list = [bit_matrix.copy()]
        pm_list = [0.0]
        split_pos = self.information_pos.tolist()
        split_loc = 0
        l_now = 1
        frozen_val = 0

        while split_loc < len(split_pos):
            for i in range(l_now):
                llr_m = llr_list[i]
                bit_m = bit_list[i]
                pm0 = pm_list[i]
                llr_m, bit_m = sc_stepping_decoder(
                    llr_m,
                    bit_m,
                    self.information_pos,
                    frozen_val,
                    split_pos[split_loc],
                )
                llr_list[i] = llr_m
                bit_list[i] = bit_m

                start = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                end = split_pos[split_loc] + 1
                seg_llr = llr_m[n][start:end]
                seg_bit = bit_m[n][start:end]
                pm_list[i] = pm0 + _pm_update_segment(seg_llr, seg_bit)

                if self.frozen_bits[split_pos[split_loc]] == 0:
                    llr_list.append(llr_m.copy())
                    bit_w = bit_m.copy()
                    bit_w[n][split_pos[split_loc]] = 1 - bit_w[n][split_pos[split_loc]]
                    bit_list.append(bit_w)
                    seg_bit_w = bit_w[n][start:end]
                    pm_list.append(pm0 + _pm_update_segment(seg_llr, seg_bit_w))

            if l_now * 2 > self.list_size:
                order = np.argsort(pm_list)
                keep = order[: self.list_size]
                pm_list = [pm_list[i] for i in keep]
                llr_list = [llr_list[i] for i in keep]
                bit_list = [bit_list[i] for i in keep]
            l_now = len(pm_list)
            split_loc += 1

        if len(split_pos) > 0 and split_pos[-1] != N - 1:
            for i in range(l_now):
                llr_m, bit_m = sc_stepping_decoder(
                    llr_list[i],
                    bit_list[i],
                    self.information_pos,
                    frozen_val,
                    N - 1,
                )
                llr_list[i] = llr_m
                bit_list[i] = bit_m
                start = split_pos[-1] + 1
                pm_list[i] += _pm_update_segment(llr_m[n][start:N], bit_m[n][start:N])

        order = np.argsort(pm_list)
        best_u = None
        best_pm = None
        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][n].astype(int)
                payload = u_cand[self.information_pos]
                if crc_check(payload, self.crc_length):
                    best_u = u_cand
                    best_pm = pm_list[idx]
                    break
        if best_u is None:
            idx = order[0]
            best_u = bit_list[idx][n].astype(int)
            best_pm = pm_list[idx]

        return best_u.copy(), float(best_pm)
