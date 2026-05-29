"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import copy
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import sc_decode, _bit_reversed, _active_llr_level, _active_bit_level, _upper_llr, _lower_llr


_CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode_bits(info_bits, crc_length=8):
    """CRC 编码"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC_POLYS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)

    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask

    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check_bits(bits, crc_length=8):
    """CRC 校验"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC_POLYS[crc_length]
    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)

    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg == 0


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


def _update_llrs_path(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = _lower_llr(
                    L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                )


def _update_bits_path(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class SCLDecoder:
    """SCL 译码器（路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_u = llr_ch[self.br]
        N, n = self.N, self.n

        paths = []
        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_u
        paths.append({"L": L0, "B": B0, "u": np.zeros(N, dtype=int), "pm": 0.0})

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                L, B, u, pm = path["L"], path["B"], path["u"], path["pm"]
                _update_llrs_path(L, B, l, n, N)
                llr0 = L[l, n]

                if l in self.frozen_set:
                    L2 = copy.deepcopy(L)
                    B2 = copy.deepcopy(B)
                    u2 = u.copy()
                    u2[l] = 0
                    B2[l, n] = 0
                    _update_bits_path(B2, l, n, N)
                    new_paths.append({
                        "L": L2, "B": B2, "u": u2,
                        "pm": _pm_update(pm, llr0, 0),
                    })
                else:
                    for bit in (0, 1):
                        L2 = copy.deepcopy(L)
                        B2 = copy.deepcopy(B)
                        u2 = u.copy()
                        u2[l] = bit
                        B2[l, n] = bit
                        _update_bits_path(B2, l, n, N)
                        new_paths.append({
                            "L": L2, "B": B2, "u": u2,
                            "pm": _pm_update(pm, llr0, bit),
                        })

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check_bits(p["u"][self.info_indices], self.crc_length)]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p["pm"])
        return best["u"].copy(), best["pm"]


crc_encode = crc_encode_bits
crc_check = crc_check_bits
