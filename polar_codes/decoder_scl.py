"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _lazy_llr,
    _update_partial_sum,
    path_metric_update,
    sc_llr_to_bit,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(
        bits[-crc_length:],
        crc_encode(bits[:-crc_length], crc_length)[-crc_length:],
    )


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _new_path(self, llr_ch):
        llrs = np.full((self.n + 1, self.N), -np.inf, dtype=np.float64)
        llrs[self.n, :] = llr_ch
        return {
            "llrs": llrs,
            "s": np.full((self.n + 1, self.N), -1, dtype=int),
            "u_hat": np.zeros(self.N, dtype=int),
            "pm": 0.0,
        }

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                path["llrs"] = np.full((self.n + 1, self.N), -np.inf, dtype=np.float64)
                path["llrs"][self.n, :] = llr_ch

                if self.frozen_bits[phi]:
                    llr_val = _lazy_llr(0, phi, path["llrs"], path["s"], self.n)
                    pm = path_metric_update(path["pm"], llr_val, 0)
                    new_path = {
                        "llrs": path["llrs"],
                        "s": path["s"].copy(),
                        "u_hat": path["u_hat"].copy(),
                        "pm": pm,
                    }
                    new_path["u_hat"][phi] = 0
                    new_path["s"][0, phi] = 0
                    new_path["llrs"][0, phi] = np.inf
                    candidates.append(new_path)
                else:
                    llr_val = _lazy_llr(0, phi, path["llrs"], path["s"], self.n)
                    for bit in (0, 1):
                        pm = path_metric_update(path["pm"], llr_val, bit)
                        new_path = {
                            "llrs": path["llrs"].copy(),
                            "s": path["s"].copy(),
                            "u_hat": path["u_hat"].copy(),
                            "pm": pm,
                        }
                        new_path["u_hat"][phi] = bit
                        new_path["s"][0, phi] = bit
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
