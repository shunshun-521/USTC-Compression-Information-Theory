"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _frozen_indices_from_mask,
    _update_bits,
    _update_llrs,
    f_operation,
)


def _pm_penalty(llr, u):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u == preferred else abs(llr)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")
    reg = 0
    for bit in info_bits:
        reg ^= bit << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)
    return np.array_equal(expected, bits)


class SCLDecoder:
    """SCL 译码器（路径复制实现，适用于中等列表大小）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = _frozen_indices_from_mask(frozen_bits)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(
            sorted(set(range(N)) - self.frozen_set), dtype=int
        )

    def _path_llr(self, path, l):
        return path["L"][l, self.n]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        base_L = np.zeros((N, n + 1), dtype=np.float64)
        base_B = np.zeros((N, n + 1), dtype=np.float64)
        base_L[:, 0] = llr_ch
        paths = [{"L": base_L, "B": base_B, "pm": 0.0}]

        for phi in range(N):
            l = _bit_reversed(phi, n)
            candidates = []
            for path in paths:
                _update_llrs(path["L"], path["B"], l, n, N)
                llr = path["L"][l, n]
                if l in self.frozen_set:
                    child = path
                    child["pm"] = child["pm"] + _pm_penalty(llr, 0)
                    child["B"][l, n] = 0
                    _update_bits(child["B"], l, n, N)
                    candidates.append(child)
                else:
                    for u in (0, 1):
                        child = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": path["pm"] + _pm_penalty(llr, u),
                        }
                        child["B"][l, n] = u
                        _update_bits(child["B"], l, n, N)
                        candidates.append(child)
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path["B"][:, n].astype(int)
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, n].astype(int), best["pm"]
