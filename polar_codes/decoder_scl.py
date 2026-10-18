"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    bit_reversed,
    active_llr_level,
    active_bit_level,
    upper_llr,
    lower_llr,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, degree):
    mask = (1 << degree) - 1
    reg = 0
    for b in bits:
        reg ^= int(b) << (degree - 1)
        for _ in range(degree):
            if reg & (1 << (degree - 1)):
                reg = ((reg << 1) & mask) ^ poly
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        rem = _crc_remainder(bits, _CRC8_POLY, 8)
        return rem == 0
    if crc_length == 16:
        rem = _crc_remainder(bits, _CRC16_POLY, 16)
        return rem == 0
    raise ValueError("crc_length must be 8 or 16")


def _update_llrs_path(l, L, B, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = int(B[j - branch_size, s + 1])
                L[j, s + 1] = lower_llr(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits_path(l, B, n):
    if l < B.shape[0] / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组引用）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "pm": 0.0,
            "u": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                _update_llrs_path(l, path["L"], path["B"], n)
                llr = path["L"][l, n]

                if l in self.frozen_set:
                    u_choice = [0]
                else:
                    u_choice = [0, 1]

                for u_bit in u_choice:
                    p = {
                        "L": path["L"],
                        "B": path["B"].copy(),
                        "pm": path["pm"] + self._pm_penalty(llr, u_bit),
                        "u": path["u"].copy(),
                    }
                    p["B"][l, n] = u_bit
                    p["u"][l] = u_bit
                    _update_bits_path(l, p["B"], n)
                    new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        # 选择最优路径
        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            info_idx = np.where(info_mask)[0]
            valid = []
            for p in paths:
                bits = p["u"][info_idx]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
