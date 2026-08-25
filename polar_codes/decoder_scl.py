"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed_index,
    f_operation,
    g_operation,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_divide(bits, crc_length):
    """对比特序列做 CRC 除法"""
    poly = CRC_POLYNOMIALS[crc_length]
    temp = list(map(int, bits))
    for i in range(len(temp) - crc_length):
        if temp[i]:
            for j in range(crc_length + 1):
                if (poly >> (crc_length - j)) & 1:
                    temp[i + j] ^= 1
    return temp


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    padded = list(map(int, info_bits)) + [0] * crc_length
    temp = _crc_divide(padded, crc_length)
    return np.array(list(map(int, info_bits)) + temp[-crc_length:], dtype=int)


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    remainder = _crc_divide(bits, crc_length)
    return sum(remainder[-crc_length:]) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)
        self.decode_order = [bit_reversed_index(phi, self.n) for phi in range(N)]

    @staticmethod
    def _pm_update(pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        N = self.N
        n = self.n
        L_size = self.list_size
        llr = np.asarray(llr_ch, dtype=np.float64)[self.rev]

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=np.int8),
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr

        for phi in range(N):
            l = self.decode_order[phi]
            is_frozen = l in self.frozen_set
            candidates = []

            for path in paths:
                L = path["L"].copy()
                B = path["B"].copy()
                self._update_llrs(L, B, l)
                cur_llr = L[l, n]

                if is_frozen:
                    u = 0
                    B[l, n] = u
                    new_u = path["u_hat"].copy()
                    new_u[l] = u
                    self._update_bits(B, l)
                    candidates.append(
                        {
                            "pm": self._pm_update(path["pm"], cur_llr, u),
                            "L": L,
                            "B": B,
                            "u_hat": new_u,
                        }
                    )
                else:
                    for u in (0, 1):
                        Lc = L.copy()
                        Bc = B.copy()
                        Lc[l, n] = cur_llr
                        Bc[l, n] = u
                        new_u = path["u_hat"].copy()
                        new_u[l] = u
                        self._update_bits(Bc, l)
                        candidates.append(
                            {
                                "pm": self._pm_update(path["pm"], cur_llr, u),
                                "L": Lc,
                                "B": Bc,
                                "u_hat": new_u,
                            }
                        )

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:L_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][~self.frozen_bits], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"], best["pm"]
