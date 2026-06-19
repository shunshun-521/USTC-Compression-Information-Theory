"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    _upper_llr,
    _lower_llr,
    _bit_reverse,
    _active_llr_level,
    _active_bit_level,
    precompute_sc_indices,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << crc_length
    for bit in bits:
        reg <<= 1
        if bit:
            reg |= 1
        if reg & top:
            reg ^= poly
    return reg & mask


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = bits[-crc_length:]
    remainder = _crc_remainder(payload, poly, crc_length)
    expected_val = 0
    for b in expected:
        expected_val = (expected_val << 1) | int(b)
    return remainder == expected_val


def _pm_penalty(llr, u_bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr)


def _advance_llr_path(L, B, l, n):
    start_s = n - _active_llr_level(l, n)
    N = L.shape[0]
    for s in range(start_s, n):
        block_size = 1 << (s + 1)
        branch_size = block_size >> 1
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                )


def _update_bits_path(B, l, n):
    if l < (1 << (n - 1)):
        return
    N = B.shape[0]
    start_b = n - _active_bit_level(l, n)
    for s in range(n, start_b, -1):
        block_size = 1 << s
        branch_size = block_size >> 1
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（copy-on-split 路径管理）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = []
        L0 = np.zeros((N, n + 1), dtype=np.float64)
        B0 = np.zeros((N, n + 1), dtype=np.int8)
        L0[:, 0] = llr_ch
        paths.append({"L": L0, "B": B0, "pm": 0.0, "u": np.zeros(N, dtype=np.int8)})

        for i in range(N):
            l = _bit_reverse(i, n)
            candidates = []

            for path in paths:
                _advance_llr_path(path["L"], path["B"], l, n)
                cur_llr = path["L"][l, n]

                if self.frozen_bits[l]:
                    pm = path["pm"] + _pm_penalty(cur_llr, 0)
                    new_L = path["L"].copy()
                    new_B = path["B"].copy()
                    new_u = path["u"].copy()
                    new_B[l, n] = 0
                    new_u[l] = 0
                    _update_bits_path(new_B, l, n)
                    candidates.append({"L": new_L, "B": new_B, "pm": pm, "u": new_u})
                else:
                    for u_bit in (0, 1):
                        pm = path["pm"] + _pm_penalty(cur_llr, u_bit)
                        new_L = path["L"].copy()
                        new_B = path["B"].copy()
                        new_u = path["u"].copy()
                        new_B[l, n] = u_bit
                        new_u[l] = u_bit
                        _update_bits_path(new_B, l, n)
                        candidates.append({"L": new_L, "B": new_B, "pm": pm, "u": new_u})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u"].astype(int), best["pm"]
