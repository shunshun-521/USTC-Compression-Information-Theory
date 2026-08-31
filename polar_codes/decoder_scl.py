"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError(f"Unsupported CRC length: {crc_length}")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    @staticmethod
    def _pm_penalty(llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    child = {
                        "pm": path["pm"] + self._pm_penalty(llr, 0),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    child["B"][l, n] = 0
                    _update_bits(child["B"], l, n, N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = {
                            "pm": path["pm"] + self._pm_penalty(llr, u),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        child["B"][l, n] = u
                        _update_bits(child["B"], l, n, N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        crc_valid = [p for p in paths if crc_check(p["B"][:, n], self.crc_length)]
        best = min(crc_valid if crc_valid else paths, key=lambda p: p["pm"])

        return best["B"][:, n].astype(int), best["pm"]
