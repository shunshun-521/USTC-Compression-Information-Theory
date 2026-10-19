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
    f_operation,
    g_operation,
    precompute_sc_indices,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_remainder(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 L/B 阵列）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        _, self.llr_layer_vec, self.bit_layer_vec = precompute_sc_indices(N)
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[bit_reversal_permutation(self.N)]

        paths = []
        L0 = np.zeros((self.N, self.m + 1), dtype=np.float64)
        B0 = np.zeros((self.N, self.m + 1), dtype=np.int8)
        L0[:, 0] = llr_ch
        paths.append({"L": L0, "B": B0, "pm": 0.0, "u": np.zeros(self.N, dtype=np.int32)})

        for i in range(self.N):
            l = _bit_reversed(i, self.m)
            new_paths = []

            for p in paths:
                L, B = p["L"], p["B"]
                _update_llrs(L, B, l, self.m)
                llr_leaf = L[l, self.m]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_leaf, 0)
                    p["pm"] += pen
                    p["u"][l] = 0
                    B[l, self.m] = 0
                    _update_bits(B, l, self.m)
                    new_paths.append(p)
                else:
                    for ubit in (0, 1):
                        cp = {
                            "L": L.copy(),
                            "B": B.copy(),
                            "pm": p["pm"] + self._pm_penalty(llr_leaf, ubit),
                            "u": p["u"].copy(),
                        }
                        cp["u"][l] = ubit
                        cp["B"][l, self.m] = ubit
                        _update_bits(cp["B"], l, self.m)
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            chosen = min(valid, key=lambda x: x["pm"]) if valid else min(
                paths, key=lambda x: x["pm"]
            )
        else:
            chosen = min(paths, key=lambda x: x["pm"])

        return chosen["u"].copy(), chosen["pm"]
