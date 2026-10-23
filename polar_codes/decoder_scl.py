"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
    f_operation,
    g_operation,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.asarray(info_indices, dtype=int) if info_indices is not None else None

    def _pm_add(self, pm, llr, u):
        v = 0 if llr >= 0 else 1
        return pm + (0.0 if u == v else abs(llr))

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed(phi, n)
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]
                if self.frozen_bits[l]:
                    pm = self._pm_add(path["pm"], llr, 0)
                    child = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    child["B"][l, n] = 0
                    child["u_hat"][l] = 0
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = self._pm_add(path["pm"], llr, u)
                        child = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["B"][l, n] = u
                        child["u_hat"][l] = u
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]
            for path in paths:
                _update_bits(path["B"], l, n)

        if self.crc_length > 0 and self.info_indices is not None:
            valid = []
            for p in paths:
                bits = p["u_hat"][self.info_indices]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
