"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _upper_llr,
    _lower_llr,
    _active_llr_level,
    _active_bit_level,
    _bit_reversed_index,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """
    CRC 编码：将校验位附加在信息比特之后。
    r=8: 0x07 (x^8+x^2+x+1), r=16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (poly if fb else 0)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        N, n, list_sz = self.N, self.n, self.list_size

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
                "u": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        decode_order = [_bit_reversed_index(i, n) for i in range(N)]

        for l in decode_order:
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
                            L[j, s + 1] = _lower_llr(
                                L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                            )

                llr_bit = L[l, n]
                if self.frozen_bits[l]:
                    penalty = 0.0 if llr_bit >= 0 else abs(llr_bit)
                    cp = self._copy_path(path)
                    cp["pm"] += penalty
                    cp["u"][l] = 0
                    cp["B"][l, n] = 0
                    self._update_bits(cp, l)
                    new_paths.append(cp)
                else:
                    for bit in (0, 1):
                        penalty = 0.0 if (bit == 0 and llr_bit >= 0) or (bit == 1 and llr_bit < 0) else abs(llr_bit)
                        cp = self._copy_path(path)
                        cp["pm"] += penalty
                        cp["u"][l] = bit
                        cp["B"][l, n] = bit
                        self._update_bits(cp, l)
                        new_paths.append(cp)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:list_sz]

        best = self._select_path(paths)
        return best["u"].astype(int), best["pm"]

    def _copy_path(self, path):
        return {
            "pm": path["pm"],
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "u": path["u"].copy(),
        }

    def _update_bits(self, path, l):
        B, n, N = path["B"], self.n, self.N
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def _select_path(self, paths):
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = []
            for p in paths:
                info_bits = p["u"][info_positions]
                if len(info_bits) >= self.crc_length and crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                return min(valid, key=lambda p: p["pm"])
        return min(paths, key=lambda p: p["pm"])
