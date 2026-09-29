"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _FastSCContext


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _branch_pm(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        if u == hard:
            return pm
        return pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [
            {"pm": 0.0, "u": np.zeros(self.N, dtype=np.int8), "ctx": _FastSCContext(self.N)}
        ]

        for idx in range(self.N):
            new_paths = []
            for path in paths:
                prefix = path["u"][:idx]
                if self.frozen_bits[idx]:
                    llr = path["ctx"].fast_llr(idx, llr_ch, prefix)
                    child = {
                        "pm": self._branch_pm(path["pm"], llr, 0),
                        "u": path["u"].copy(),
                        "ctx": _FastSCContext(self.N),
                    }
                    child["u"][idx] = 0
                    new_paths.append(child)
                else:
                    llr = path["ctx"].fast_llr(idx, llr_ch, prefix)
                    for u in (0, 1):
                        child = {
                            "pm": self._branch_pm(path["pm"], llr, u),
                            "u": path["u"].copy(),
                            "ctx": _FastSCContext(self.N),
                        }
                        child["u"][idx] = u
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                info_bits = p["u"][self.info_indices]
                crc_ok.append(crc_check(info_bits, self.crc_length))
            else:
                crc_ok.append(True)

        if any(crc_ok):
            pool = [p for p, ok in zip(paths, crc_ok) if ok]
            best = min(pool, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
