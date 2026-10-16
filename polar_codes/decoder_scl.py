"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import compute_llr_for_bit


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 u_hat）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "u_hat": np.zeros(self.N, dtype=np.int8)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_phi = compute_llr_for_bit(llr_ch, path["u_hat"], phi)
                if self.frozen_bits[phi]:
                    pen = abs(llr_phi) if llr_phi < 0 else 0.0
                    new_path = copy.copy(path)
                    new_path["u_hat"] = path["u_hat"].copy()
                    new_path["pm"] = path["pm"] + pen
                    new_path["u_hat"][phi] = 0
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pen = abs(llr_phi) if (bit == 0 and llr_phi < 0) or (bit == 1 and llr_phi >= 0) else 0.0
                        new_path = {
                            "pm": path["pm"] + pen,
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["u_hat"][phi] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        crc_pass = []
        for path in paths:
            if self.crc_length > 0:
                info = path["u_hat"][self.info_indices]
                if crc_check(info, self.crc_length):
                    crc_pass.append(path)

        if self.crc_length > 0 and crc_pass:
            best = min(crc_pass, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].astype(int), float(best["pm"])
