"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _to_tree_frozen_mask,
)


CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
CRC16_POLY = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1], dtype=np.int8
)


def _crc_remainder(bits, poly):
    bits = np.asarray(bits, dtype=np.int8).copy()
    r = len(poly) - 1
    for i in range(len(bits) - r):
        if bits[i] == 1:
            bits[i : i + len(poly)] ^= poly
    return bits[-r:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)]), poly
    )
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([bits[:-crc_length], np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_remainder(padded, poly)
    return np.array_equal(remainder, bits[-crc_length:])


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.tree_frozen = _to_tree_frozen_mask(self.frozen_bits, N)
        self.frozen_set = set(np.where(self.tree_frozen)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _init_path(self, llr_perm):
        return {
            "pm": 0.0,
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int8),
        }

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        if self.list_size == 1 and self.crc_length == 0:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_perm = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        paths = [self._init_path(llr_perm)]
        paths[0]["L"][:, 0] = llr_perm

        for i in range(self.N):
            l = _bit_reversed_index(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr0 = path["L"][l, self.n]

                if l in self.frozen_set:
                    new_path = {
                        "pm": path["pm"] + (0.0 if llr0 >= 0 else abs(llr0)),
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    new_path["B"][l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = {
                            "pm": path["pm"]
                            + (
                                0.0
                                if (bit == 0 and llr0 >= 0) or (bit == 1 and llr0 < 0)
                                else abs(llr0)
                            ),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        new_path["B"][l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                u_hat = path["B"][:, self.n].astype(int)
                if crc_check(u_hat[self.info_indices], self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["B"][:, self.n].astype(int).copy(), best["pm"]
