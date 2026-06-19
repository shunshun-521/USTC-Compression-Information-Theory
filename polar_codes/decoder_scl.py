"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    bit_reversal_permutation,
    bit_reversed_index,
    upper_llr,
    lower_llr,
    active_llr_level,
    active_bit_level,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（MSB-first）。"""
    info_bits = np.asarray(info_bits, dtype=int)
    reg = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    return _crc_remainder(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _copy_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def _update_llrs(self, path, l):
        L = path["L"]
        B = path["B"]
        n = self.n
        N = self.N
        for s in range(n - active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s],
                        L[j - branch_size, s],
                        int(B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        L = path["L"]
        B = path["B"]
        n = self.n
        N = self.N
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_ch = llr_ch[self.br]

        n = self.n
        N = self.N

        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_ch

        paths = [
            {
                "L": L0,
                "B": B0,
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=int),
            }
        ]

        for phi in range(N):
            l = bit_reversed_index(phi, n)
            is_frozen = l in self.frozen_set
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path["L"][l, n]

                if is_frozen:
                    new_path = self._copy_path(path)
                    penalty = self._pm_penalty(llr, 0)
                    new_path["pm"] += penalty
                    new_path["B"][l, n] = 0
                    new_path["u_hat"][l] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_cand in (0, 1):
                        new_path = self._copy_path(path)
                        penalty = self._pm_penalty(llr, u_cand)
                        new_path["pm"] += penalty
                        new_path["B"][l, n] = u_cand
                        new_path["u_hat"][l] = u_cand
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            info_positions = np.where(self.frozen_bits == 0)[0]
            crc_ok = [
                p
                for p in paths
                if crc_check(p["u_hat"][info_positions], self.crc_length)
            ]
            if crc_ok:
                best = min(crc_ok, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
