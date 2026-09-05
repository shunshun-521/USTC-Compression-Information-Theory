"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
)
from encoder import bit_reversed_index


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length not in CRC_POLYNOMIALS:
        raise ValueError(f"Unsupported CRC length: {crc_length}")

    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    return np.array_equal(crc_encode(bits[:-crc_length], crc_length), bits)


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, L, B, pm, u_hat):
        self.L = L
        self.B = B
        self.pm = pm
        self.u_hat = u_hat


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径列表）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    @staticmethod
    def _llr_penalty(llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr_val)

    def _update_llrs(self, L, B, l):
        n = self.n
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

    def _update_bits(self, B, l):
        n = self.n
        N = self.N
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N

        L0 = np.full((N, n + 1), np.nan, dtype=np.float64)
        B0 = np.full((N, n + 1), np.nan)
        L0[:, 0] = llr_ch
        paths = [_PathState(L0, B0, 0.0, np.zeros(N, dtype=int))]

        for l in [bit_reversed_index(i, n) for i in range(N)]:
            candidates = []
            for path in paths:
                self._update_llrs(path.L, path.B, l)
                cur_llr = path.L[l, n]
                if l in self.frozen_set:
                    pm = path.pm + self._llr_penalty(cur_llr, 0)
                    candidates.append((pm, path, 0))
                else:
                    for u_bit in (0, 1):
                        pm = path.pm + self._llr_penalty(cur_llr, u_bit)
                        candidates.append((pm, path, u_bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u_bit in selected:
                L_copy = parent.L.copy()
                B_copy = parent.B.copy()
                u_copy = parent.u_hat.copy()
                u_copy[l] = u_bit
                B_copy[l, n] = u_bit
                self._update_bits(B_copy, l)
                new_paths.append(_PathState(L_copy, B_copy, pm, u_copy))
            paths = new_paths

        crc_pass = []
        for path in paths:
            payload = path.u_hat[self.info_indices]
            if self.crc_length == 0 or crc_check(payload, self.crc_length):
                crc_pass.append(path)

        pool = crc_pass if crc_pass else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
