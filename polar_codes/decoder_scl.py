"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _boxplus,
    _lower_llr,
    _prepare_channel_llr,
)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ poly) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=int)
    elif crc_length == 16:
        poly = 0x8005
        reg = 0xFFFF
        for bit in info_bits:
            msb = (reg >> 15) & 1
            reg = (reg << 1) & 0xFFFF
            if int(bit) ^ msb:
                reg ^= poly
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=int)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _update_llrs(self, L, B, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _boxplus(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch_size, s], int(B[j - branch_size, s + 1])
                    )
        return L[l, self.n]

    def _update_bits(self, B, l, bit):
        B[l, self.n] = bit
        if l < self.N // 2:
            return B
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
        return B

    def decode(self, llr_ch):
        llr = _prepare_channel_llr(llr_ch)
        paths = [
            {
                "L": np.full((self.N, self.n + 1), np.nan, dtype=np.float64),
                "B": np.full((self.N, self.n + 1), np.nan),
                "pm": 0.0,
                "u_hat": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            candidates = []
            for path in paths:
                llr_val = self._update_llrs(path["L"], path["B"], l)
                if self.frozen_bits[l]:
                    bits = [0]
                else:
                    bits = [0, 1]
                for bit in bits:
                    L_copy = path["L"].copy()
                    B_copy = path["B"].copy()
                    u_hat = path["u_hat"].copy()
                    pm = path["pm"]
                    hard = 0 if llr_val >= 0 else 1
                    if bit != hard:
                        pm += abs(llr_val)
                    u_hat[l] = bit
                    B_copy = self._update_bits(B_copy, l, bit)
                    candidates.append(
                        {"L": L_copy, "B": B_copy, "pm": pm, "u_hat": u_hat}
                    )
            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
