"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC_POLYNOMIALS[crc_length]
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in info_bits:
        feedback = bit ^ reg[0]
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if feedback:
            reg ^= np.array(
                [((poly >> (crc_length - 1 - i)) & 1) for i in range(crc_length)],
                dtype=np.int8,
            )
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    recomputed = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, recomputed)


class SCLDecoder:
    """
    SCL 译码器。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )

    def _update_bits(self, B, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def _path_penalty(self, llr, bit):
        hard = hard_decision(llr)
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{
            "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
            "B": np.full((self.N, self.n + 1), np.nan),
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l)
                llr = path["L"][l, self.n]
                bits_to_try = [0] if l in self.frozen_set else [0, 1]

                for bit in bits_to_try:
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": path["pm"] + self._path_penalty(llr, bit),
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, self.n] = bit
                    new_path["u_hat"][l] = bit
                    self._update_bits(new_path["B"], l)
                    candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
