"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_llr_at_bit


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（基于 SC 逐比特 LLR 的路径扩展）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        if self.list_size == 1 and self.crc_length == 0:
            from decoder_sc import sc_decode

            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0

        paths = [{"pm": 0.0, "u_hat": np.zeros(N, dtype=int)}]

        for phi in range(N):
            candidates = []
            for path in paths:
                llr = sc_llr_at_bit(llr_ch, self.frozen_bits, path["u_hat"], phi)
                if self.frozen_bits[phi]:
                    u = 0
                    penalty = abs(llr) if llr < 0 else 0.0
                    candidates.append((path["pm"] + penalty, path, u))
                else:
                    for u in (0, 1):
                        consistent = (u == 0 and llr >= 0) or (u == 1 and llr < 0)
                        penalty = 0.0 if consistent else abs(llr)
                        candidates.append((path["pm"] + penalty, path, u))

            candidates.sort(key=lambda x: x[0])
            new_paths = []
            for pm, parent, u in candidates[: self.list_size]:
                u_hat = parent["u_hat"].copy()
                u_hat[phi] = u
                new_paths.append({"pm": pm, "u_hat": u_hat})
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u_hat"][self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
