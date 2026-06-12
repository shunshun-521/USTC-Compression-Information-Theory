"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _update_bits, _update_llrs, sc_decode
from encoder import bit_reversed


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 尾部 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_update(pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if hard != bit:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [
            {
                "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
                "pm": 0.0,
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for step in range(self.N):
            leaf = bit_reversed(step, self.n)
            candidates = []

            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, leaf, self.n)
                llr = L[leaf, self.n]

                if self.frozen_bits[leaf]:
                    candidates.append((self._pm_update(path["pm"], llr, 0), path, 0))
                else:
                    for bit in (0, 1):
                        candidates.append(
                            (self._pm_update(path["pm"], llr, bit), path, bit)
                        )

            candidates.sort(key=lambda item: item[0])
            new_paths = []
            for pm, parent, bit in candidates[: self.list_size]:
                child = {
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "pm": pm,
                }
                child["B"][leaf, self.n] = bit
                _update_bits(child["B"], leaf, self.n, self.N)
                new_paths.append(child)
            paths = new_paths

        paths.sort(key=lambda p: p["pm"])

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path["B"][:, self.n].astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = paths[0]
        return best["B"][:, self.n].astype(int), best["pm"]
