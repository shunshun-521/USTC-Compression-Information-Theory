"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位附加"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（PSC 结构 + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]
        self.br = bit_reversal_permutation(N)

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, B, l, u_bit):
        B[l, self.n] = u_bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_internal = llr_ch[self.br]

        paths = []
        L0 = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B0 = np.zeros((self.N, self.n + 1), dtype=np.int8)
        L0[:, 0] = llr_internal
        paths.append({"L": L0, "B": B0, "pm": 0.0, "u": np.zeros(self.N, dtype=np.int8)})

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr_bit = path["L"][l, self.n]

                if l in self.frozen_set:
                    pm = path["pm"] + self._path_metric_penalty(llr_bit, 0)
                    Ln = path["L"].copy()
                    Bn = path["B"].copy()
                    un = path["u"].copy()
                    un[l] = 0
                    self._update_bits(Bn, l, 0)
                    candidates.append({"L": Ln, "B": Bn, "pm": pm, "u": un})
                else:
                    for u_cand in (0, 1):
                        pm = path["pm"] + self._path_metric_penalty(llr_bit, u_cand)
                        Ln = path["L"].copy()
                        Bn = path["B"].copy()
                        un = path["u"].copy()
                        un[l] = u_cand
                        self._update_bits(Bn, l, u_cand)
                        candidates.append({"L": Ln, "B": Bn, "pm": pm, "u": un})

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        paths.sort(key=lambda p: p["pm"])
        if self.crc_length > 0:
            passed = [
                p
                for p in paths
                if crc_check(p["u"][self.info_positions], self.crc_length)
            ]
            best = passed[0] if passed else paths[0]
        else:
            best = paths[0]

        return best["u"].astype(int), best["pm"]
