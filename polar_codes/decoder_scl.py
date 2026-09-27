"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _bit_reversed,
    _prepare_channel_llr,
    sc_update_bits_for_bit,
    sc_update_llrs_for_bit,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).reshape(-1)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).reshape(-1)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


class SCLDecoder:
    """SCL 译码器（路径级 Lazy Copy：分裂时复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.bit_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        N, n = self.N, self.n

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.float64),
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=np.int32),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.bit_order:
            new_paths = []
            for path in paths:
                llr = sc_update_llrs_for_bit(path["L"], path["B"], l, n)
                if l in self.frozen_set:
                    pen = self._pm_penalty(llr, 0)
                    path["pm"] += pen
                    path["u_hat"][l] = 0
                    path["B"][l, n] = 0
                    sc_update_bits_for_bit(path["B"], l, n)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p2 = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + self._pm_penalty(llr, u),
                            "u_hat": path["u_hat"].copy(),
                        }
                        p2["u_hat"][l] = u
                        p2["B"][l, n] = u
                        sc_update_bits_for_bit(p2["B"], l, n)
                        new_paths.append(p2)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"][self.info_indices], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
