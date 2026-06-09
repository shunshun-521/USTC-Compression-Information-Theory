"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07, CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
        width = 8
    elif crc_length == 16:
        poly = 0x8005
        width = 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)

    crc_bits = np.array([(reg >> i) & 1 for i in range(width - 1, -1, -1)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length)[-crc_length:], bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]

    def _new_path(self, parent=None):
        path = {
            "pm": 0.0,
            "L": None,
            "B": None,
            "u_hat": np.zeros(self.N, dtype=np.int8),
            "parent": parent,
        }
        if parent is None:
            path["L"] = np.zeros((self.N, self.n + 1), dtype=np.float64)
            path["B"] = np.zeros((self.N, self.n + 1), dtype=np.int8)
        return path

    def _materialize(self, path):
        if path["L"] is not None:
            return path
        parent = path["parent"]
        self._materialize(parent)
        path["L"] = parent["L"].copy()
        path["B"] = parent["B"].copy()
        path["parent"] = None
        return path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        root = self._new_path()
        root["L"][:, 0] = llr_ch
        active = [root]

        for l in self.decode_order:
            candidates = []
            for path in active:
                path = self._materialize(path)
                _update_llrs(path["L"], path["B"], l, n, N)
                llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    penalty = 0.0 if llr >= 0 else abs(llr)
                    path["pm"] += penalty
                    path["u_hat"][l] = 0
                    path["B"][l, n] = 0
                    _update_bits(path["B"], l, n, N)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        child = self._new_path(parent=path)
                        child["pm"] = path["pm"]
                        child["u_hat"] = path["u_hat"].copy()
                        child["u_hat"][l] = bit
                        expected = 0 if llr >= 0 else 1
                        if bit != expected:
                            child["pm"] += abs(llr)
                        candidates.append(child)

            candidates.sort(key=lambda p: p["pm"])
            active = candidates[: self.list_size]

            for path in active:
                path = self._materialize(path)
                path["B"][l, n] = path["u_hat"][l]
                _update_bits(path["B"], l, n, N)

        active.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = [
                p
                for p in active
                if crc_check(p["u_hat"][info_mask], self.crc_length)
            ]
            best = valid[0] if valid else active[0]
        else:
            best = active[0]

        return best["u_hat"].astype(int), best["pm"]
