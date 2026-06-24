"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
    sc_decode,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    """MSB-first CRC 余数（CRC-8: 0x07, CRC-16: 0x8005）。"""
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask)
        if feedback:
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int_)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_remainder(info_bits, poly, crc_length)
    # 追加 CRC 位使整体余数为零
    extended = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int_)])
    for i in range(crc_length):
        feedback = ((reg >> (crc_length - 1)) & 1)
        reg = ((reg << 1) & ((1 << crc_length) - 1))
        if feedback:
            reg ^= poly
        extended[len(info_bits) + i] = feedback
    return extended


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int_)
    if crc_length == 0:
        return True
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径独立 L/B 矩阵）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=np.int_)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.info_idx = np.where(self.frozen_bits == 0)[0]

    def _pm_update(self, pm, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        if u != u_from_llr:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        N, n = self.N, self.n
        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int_),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []
            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                _update_llrs(L, B, l, n)
                cur_llr = L[l, n]

                if l in self.frozen_set:
                    candidates = [0]
                else:
                    candidates = [0, 1]

                for u in candidates:
                    B2 = B.copy()
                    B2[l, n] = u
                    _update_bits(B2, l, n)
                    new_paths.append({
                        "pm": self._pm_update(path["pm"], cur_llr, u),
                        "L": L,
                        "B": B2,
                    })

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            passing = []
            for p in paths:
                u_hat = p["B"][:, n].astype(int)
                info_bits = u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    passing.append(p)
            if passing:
                best = min(passing, key=lambda p: p["pm"])

        u_hat = best["B"][:, n].astype(int)
        return u_hat, best["pm"]
