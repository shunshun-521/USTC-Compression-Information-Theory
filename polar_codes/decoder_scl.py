"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import copy
from encoder import bit_reversal_permutation
from decoder_sc import (
    sc_decode,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    upper_llr,
    lower_llr,
    _hard_decision,
)


def crc_encode(info_bits, crc_length=8):
    """CRC 校验位计算并附加"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
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
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_idx = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _pm_penalty(self, llr_val, u_bit):
        preferred = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == preferred else abs(llr_val)

    def _update_llrs(self, L, B, l):
        N, n = self.N, self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = llr_ch[self.rev]
        N, n = self.N, self.n

        paths = []
        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_perm
        paths.append({"pm": 0.0, "L": L0, "B": B0})

        for i in range(N):
            l = _bit_reversed(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr_leaf = path["L"][l, n]

                if l in self.frozen_idx:
                    pen = self._pm_penalty(llr_leaf, 0)
                    path["pm"] += pen
                    path["B"][l, n] = 0
                    self._update_bits(path["B"], l)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        fork = {
                            "pm": path["pm"] + self._pm_penalty(llr_leaf, u_bit),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        fork["B"][l, n] = u_bit
                        self._update_bits(fork["B"], l)
                        new_paths.append(fork)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:self.list_size]

        best_pm = float("inf")
        best_u = None
        crc_candidates = []

        for path in paths:
            u_hat = path["B"][:, n].astype(int)
            if self.crc_length > 0:
                info_idx = np.where(~self.frozen_bits)[0]
                payload = u_hat[info_idx]
                if crc_check(payload, self.crc_length):
                    crc_candidates.append((path["pm"], u_hat))
            if path["pm"] < best_pm:
                best_pm = path["pm"]
                best_u = u_hat

        if crc_candidates:
            crc_candidates.sort(key=lambda x: x[0])
            return crc_candidates[0][1], crc_candidates[0][0]

        return best_u, best_pm
