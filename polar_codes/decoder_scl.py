"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
    _permute_llr_for_decode,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
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
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（列表路径管理）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = {i for i in range(N) if self.frozen_bits[i] == 1}
        self.info_indices = np.where(self.frozen_bits == 0)[0]

    def _new_path(self, llr_perm):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_perm.copy()
        return {"L": L, "B": B, "pm": 0.0, "u_hat": np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
        }

    def _advance_llr(self, path, l):
        L, B = path["L"], path["B"]
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )
        return L[l, n]

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N / 2:
            return
        n = self.n
        N = self.N
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_perm = _permute_llr_for_decode(llr_ch, self.N)
        paths = [self._new_path(llr_perm)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                llr_val = self._advance_llr(path, l)

                if l in self.frozen_set:
                    penalty = 0.0 if llr_val >= 0 else abs(llr_val)
                    path["pm"] += penalty
                    path["u_hat"][l] = 0
                    path["B"][l, self.n] = 0
                    self._update_bits(path, l)
                    new_paths.append(path)
                else:
                    for u_val in (0, 1):
                        child = self._copy_path(path)
                        penalty = 0.0 if (
                            (u_val == 0 and llr_val >= 0) or (u_val == 1 and llr_val < 0)
                        ) else abs(llr_val)
                        child["pm"] += penalty
                        child["u_hat"][l] = u_val
                        child["B"][l, self.n] = u_val
                        self._update_bits(child, l)
                        new_paths.append(child)

            if len(new_paths) > self.list_size:
                new_paths.sort(key=lambda p: p["pm"])
                new_paths = new_paths[:self.list_size]
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"], best["pm"]
