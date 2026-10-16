"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import LARGE, _llr_for_bit, f_operation, g_operation, precompute_sc_indices

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


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
    info_bits = np.asarray(info_bits, dtype=np.int64).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int64)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int64).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _path_metric_penalty(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


class SCLDecoder:
    """SCL 译码器（路径独立维护部分 u_hat）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        precompute_sc_indices(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"u_hat": np.zeros(self.N, dtype=np.int64), "pm": 0.0}]

        for phi in range(self.N):
            new_paths = []
            for path in paths:
                metric = _llr_for_bit(phi, llr_ch, self.frozen_bits, path["u_hat"])
                if self.frozen_bits[phi]:
                    pen = _path_metric_penalty(metric, 0)
                    child = {
                        "u_hat": path["u_hat"].copy(),
                        "pm": path["pm"] + pen,
                    }
                    child["u_hat"][phi] = 0
                    new_paths.append(child)
                else:
                    for u_bit in (0, 1):
                        child = {
                            "u_hat": path["u_hat"].copy(),
                            "pm": path["pm"] + _path_metric_penalty(metric, u_bit),
                        }
                        child["u_hat"][phi] = u_bit
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        info_mask = ~self.frozen_bits
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p["u_hat"][info_mask]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = min(valid, key=lambda p: p["pm"]) if valid else paths[0]
        else:
            best = paths[0]

        return best["u_hat"].copy(), best["pm"]
