"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy

import numpy as np

from decoder_sc import (
    _SCD,
    _bit_reversed,
    clip_llr,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（基于置换 SC 状态的路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = clip_llr(np.asarray(llr_ch, dtype=np.float64))
        paths = [{"pm": 0.0, "scd": _SCD(self.N, self.n, llr_ch, self.frozen_set)}]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                scd = path["scd"]
                scd._update_llrs(l)
                llr = scd.L[l, scd.n]

                if l in self.frozen_set:
                    scd.B[l, scd.n] = 0
                    scd._update_bits(l)
                    candidates.append({
                        "pm": path["pm"] + self._pm_penalty(llr, 0),
                        "scd": scd,
                    })
                else:
                    for u in (0, 1):
                        child_scd = copy.deepcopy(scd)
                        child_scd.B[l, child_scd.n] = u
                        child_scd._update_bits(l)
                        candidates.append({
                            "pm": path["pm"] + self._pm_penalty(llr, u),
                            "scd": child_scd,
                        })

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["scd"].B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p["pm"]) if valid else paths[0]
        else:
            best = paths[0]

        u_hat = best["scd"].B[:, self.n].astype(int)
        return u_hat, best["pm"]
