"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
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
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br_idx = bit_reversal_permutation(N)
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        penalty = 0.0 if bit == hard else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [{
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=int),
            "pm": 0.0,
            "u_perm": np.zeros(self.N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                _update_llrs(path["L"], path["B"], l, self.n, self.N)
                llr = path["L"][l, self.n]
                nat_l = _bit_reversed(l, self.n)

                if nat_l in self.frozen_set:
                    new_path = self._lazy_copy(path)
                    new_path["pm"] = self._pm_update(path["pm"], llr, 0)
                    new_path["u_perm"][l] = 0
                    new_path["B"][l, self.n] = 0
                    _update_bits(new_path["B"], l, self.n, self.N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._lazy_copy(path)
                        new_path["pm"] = self._pm_update(path["pm"], llr, bit)
                        new_path["u_perm"][l] = bit
                        new_path["B"][l, self.n] = bit
                        _update_bits(new_path["B"], l, self.n, self.N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = self._select_best_path(paths)
        u_hat = best["u_perm"][self.br_idx]
        return u_hat, best["pm"]

    def _lazy_copy(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_perm": path["u_perm"].copy(),
        }

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            valid = []
            for path in paths:
                u = path["u_perm"][self.br_idx]
                payload = u[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                return min(valid, key=lambda p: p["pm"])
        return min(paths, key=lambda p: p["pm"])
