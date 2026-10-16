"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import bit_llr_at_phase, sc_decode

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bits(info_bits, crc_length):
    if crc_length == 8:
        reg = 0
        for b in info_bits:
            reg ^= int(b) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) & 0xFF) ^ CRC8_POLY
                else:
                    reg = (reg << 1) & 0xFF
        return np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) & 0xFFFF) ^ CRC16_POLY
            else:
                reg = (reg << 1) & 0xFFFF
    return np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    return np.concatenate([info_bits, _crc_bits(info_bits, crc_length)])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（路径复制 + 逐比特 LLR）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "u": np.zeros(self.N, dtype=int)}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                llr = bit_llr_at_phase(llr_ch, path["u"], phi)
                if self.frozen_bits[phi]:
                    u = 0
                    cand = {
                        "pm": path["pm"] + self._pm_penalty(llr, u),
                        "u": path["u"].copy(),
                    }
                    cand["u"][phi] = u
                    new_paths.append(cand)
                else:
                    for u in (0, 1):
                        cand = {
                            "pm": path["pm"] + self._pm_penalty(llr, u),
                            "u": path["u"].copy(),
                        }
                        cand["u"][phi] = u
                        new_paths.append(cand)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, paths[0]["pm"]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u"][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p["pm"])
        return best["u"].copy(), best["pm"]
