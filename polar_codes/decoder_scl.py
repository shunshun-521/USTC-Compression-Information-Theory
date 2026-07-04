"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from encoder import bit_reversal_permutation
from sc_core import get_pm_update, sc_decode_tree, sc_step_to_bit


def _crc_remainder(bits, poly, crc_len):
    reg = [0] * crc_len
    for bit in bits:
        fb = bit ^ reg[0]
        reg = reg[1:] + [0]
        if fb:
            for i in range(crc_len):
                if (poly >> (crc_len - 1 - i)) & 1:
                    reg[i] ^= fb
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07 (x^8+x^2+x+1)
    CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int).tolist()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits + [0] * crc_length, poly, crc_length)
    return np.array(info_bits + remainder, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits 是否包含正确的 CRC"""
    bits = np.asarray(bits, dtype=int).tolist()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    if len(bits) < crc_length:
        return False
    info = bits[:-crc_length]
    remainder = _crc_remainder(info + [0] * crc_length, poly, crc_length)
    return remainder == bits[-crc_length:]


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _prepare_llr(self, llr_ch):
        return np.asarray(llr_ch, dtype=np.float64)[self.br]

    def decode(self, llr_ch):
        llr_tree = self._prepare_llr(llr_ch)
        info_pos = self.info_indices.tolist()
        frozen_bit = 0

        if self.list_size == 1:
            u_hat, llr_leaf = sc_decode_tree(llr_tree, info_pos, frozen_bit)
            pm = get_pm_update(
                llr_leaf[info_pos],
                u_hat[info_pos].astype(int),
            )
            return u_hat.astype(int), pm

        n = self.n
        N = self.N
        llr_matrix = np.full((n + 1, N), np.nan)
        bit_matrix = np.full((n + 1, N), np.nan)
        llr_matrix[0] = llr_tree

        paths = [{"llr": llr_matrix, "bit": bit_matrix, "pm": 0.0}]
        prev_info = -1

        for bit_idx in range(N):
            new_paths = []
            for path in paths:
                llr_m = path["llr"]
                bit_m = path["bit"]
                pm0 = path["pm"]

                if self.frozen_bits[bit_idx]:
                    llr_m, bit_m = sc_step_to_bit(
                        llr_m.copy(), bit_m.copy(), info_pos, frozen_bit, bit_idx
                    )
                    llr_leaf = llr_m[n]
                    bit_leaf = bit_m[n].astype(int)
                    penalty = 0.0
                    if llr_leaf[bit_idx] < 0:
                        penalty = abs(llr_leaf[bit_idx])
                    new_paths.append(
                        {"llr": llr_m, "bit": bit_m, "pm": pm0 + penalty}
                    )
                else:
                    for bit_val in (0, 1):
                        llr_copy = llr_m.copy()
                        bit_copy = bit_m.copy()
                        llr_copy, bit_copy = sc_step_to_bit(
                            llr_copy, bit_copy, info_pos, frozen_bit, bit_idx
                        )
                        bit_copy[n, bit_idx] = bit_val
                        llr_leaf = llr_copy[n]
                        hard = 0 if llr_leaf[bit_idx] >= 0 else 1
                        penalty = 0.0 if hard == bit_val else abs(llr_leaf[bit_idx])
                        seg_start = prev_info + 1
                        seg_llr = llr_leaf[seg_start : bit_idx + 1]
                        seg_bit = bit_copy[n, seg_start : bit_idx + 1].astype(int)
                        pm_add = get_pm_update(seg_llr, seg_bit)
                        new_paths.append(
                            {
                                "llr": llr_copy,
                                "bit": bit_copy,
                                "pm": pm0 + pm_add,
                            }
                        )

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]
            if bit_idx in self.info_indices:
                prev_info = bit_idx

        for path in paths:
            u_hat = path["bit"][n].astype(int)
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    return u_hat, path["pm"]
        best = paths[0]
        return best["bit"][n].astype(int), best["pm"]
