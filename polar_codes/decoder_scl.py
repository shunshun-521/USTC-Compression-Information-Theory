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
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """CRC 校验"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    return pm + (0.0 if u == hard else abs(llr))


class SCLDecoder:
    """SCL 译码器（PSCD + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "u_hat": np.zeros(N, dtype=np.int8),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for phase in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path["L"], path["B"], phase, n, N)
                llr_val = path["L"][phase, n]

                if self.frozen_bits[phase]:
                    path["pm"] = _pm_update(path["pm"], llr_val, 0)
                    path["u_hat"][phase] = 0
                    path["B"][phase, n] = 0
                    _update_bits(path["B"], phase, n, N)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = {
                            "pm": _pm_update(path["pm"], llr_val, u_bit),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        child["u_hat"][phase] = u_bit
                        child["B"][phase, n] = u_bit
                        _update_bits(child["B"], phase, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = paths[0]
        return best["u_hat"].copy(), best["pm"]
