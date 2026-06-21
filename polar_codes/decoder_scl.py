"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _lower_llr,
    _upper_llr,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（MSB-first）。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（基于 PSCD 的列表扩展）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, L, B, l):
        n, N = self.n, self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

    def _update_bits(self, B, l):
        n, N = self.n, self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n, N = self.n, self.N

        paths = [
            {
                "pm": 0.0,
                "L": np.full((N, n + 1), np.nan, dtype=np.float64),
                "B": np.full((N, n + 1), np.nan),
            }
        ]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed_index(i, n)
            candidates = []

            for p_idx, path in enumerate(paths):
                L = path["L"].copy()
                B = path["B"].copy()
                self._update_llrs(L, B, l)
                llr_phi = L[l, n]

                if l in self.frozen_set:
                    pm = path["pm"]
                    if llr_phi < 0:
                        pm += abs(llr_phi)
                    B[l, n] = 0
                    self._update_bits(B, l)
                    candidates.append((pm, L, B))
                else:
                    for u_val in (0, 1):
                        pm = path["pm"]
                        hard = 0 if llr_phi >= 0 else 1
                        if u_val != hard:
                            pm += abs(llr_phi)
                        Bc = B.copy()
                        Bc[l, n] = u_val
                        self._update_bits(Bc, l)
                        candidates.append((pm, L, Bc))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[: self.list_size]
            paths = [{"pm": pm, "L": L, "B": B} for pm, L, B in survivors]

        best = min(paths, key=lambda p: p["pm"])
        u_hat = best["B"][:, n].astype(int)

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u = path["B"][:, n].astype(int)
                payload = u[self.info_indices]
                if crc_check(payload, self.crc_length):
                    valid.append((path["pm"], u))
            if valid:
                valid.sort(key=lambda x: x[0])
                return valid[0][1], valid[0][0]

        return u_hat, best["pm"]
