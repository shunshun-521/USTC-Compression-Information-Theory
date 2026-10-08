"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
    f_operation,
    g_operation,
)


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
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
    mask = (1 << width) - 1
    top = 1 << (width - 1)
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = [(reg >> i) & 1 for i in range(width - 1, -1, -1)]
    return np.concatenate([info_bits, crc_bits]).astype(int)


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组引用）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen)[0]

    def _path_metric_update(self, pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan, dtype=np.float64),
            "u": np.zeros(N, dtype=int),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for idx in range(N):
            l = _bit_reversed(idx, n)
            new_paths = []

            for path in paths:
                L, B, u, pm = path["L"], path["B"], path["u"], path["pm"]

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                        else:
                            top_bit = B[j - branch_size, s + 1]
                            if np.isnan(top_bit):
                                top_bit = 0
                            L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

                cur_llr = L[l, n]
                if self.frozen[l]:
                    u[l] = 0
                    B[l, n] = 0
                    new_paths.append({
                        "L": L.copy(),
                        "B": B.copy(),
                        "u": u.copy(),
                        "pm": self._path_metric_update(pm, cur_llr, 0),
                    })
                else:
                    for bit in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        uc = u.copy()
                        uc[l] = bit
                        Bc[l, n] = bit
                        new_paths.append({
                            "L": Lc,
                            "B": Bc,
                            "u": uc,
                            "pm": self._path_metric_update(pm, cur_llr, bit),
                        })

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

            for path in paths:
                l = _bit_reversed(idx, n)
                B = path["B"]
                if l >= N / 2:
                    for s in range(n, n - _active_bit_level(l, n), -1):
                        block_size = 1 << s
                        branch_size = block_size // 2
                        for j in range(l, -1, -block_size):
                            if j % block_size >= branch_size:
                                bj = 0 if np.isnan(B[j, s]) else int(B[j, s])
                                bjt = 0 if np.isnan(B[j - branch_size, s]) else int(B[j - branch_size, s])
                                B[j - branch_size, s - 1] = bj ^ bjt
                                B[j, s - 1] = B[j, s]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = p["u"][self.info_idx]
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]
