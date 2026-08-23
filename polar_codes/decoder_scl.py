"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    f_operation,
    g_operation,
    sc_decode,
)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8 (0x07) 或 CRC-16 (0x8005)
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in info_bits:
        msb = reg[0] ^ bit
        reg[:-1] = reg[1:]
        reg[-1] = 0
        if msb:
            poly_bits = [(poly >> i) & 1 for i in range(crc_length - 1, -1, -1)]
            reg ^= np.array(poly_bits, dtype=np.int8)

    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits == 1)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_penalty(self, llr, bit):
        if bit == 0:
            return 0.0 if llr >= 0 else abs(llr)
        return 0.0 if llr < 0 else abs(llr)

    def decode(self, llr_ch):
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "pm": 0.0,
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=np.int8),
            "u_hat": np.zeros(N, dtype=np.int8),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed(i, n)
            is_frozen = l in self.frozen_set
            new_paths = []

            for path in paths:
                L, B = path["L"], path["B"]

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                        else:
                            L[j, s + 1] = g_operation(
                                L[j - branch_size, s],
                                L[j, s],
                                B[j - branch_size, s + 1],
                            )

                llr_bit = L[l, n]

                if is_frozen:
                    np_path = {
                        "pm": path["pm"] + self._path_metric_penalty(llr_bit, 0),
                        "L": L.copy(),
                        "B": B.copy(),
                        "u_hat": path["u_hat"].copy(),
                    }
                    np_path["B"][l, n] = 0
                    np_path["u_hat"][l] = 0
                    self._update_bits(np_path["B"], l, n)
                    new_paths.append(np_path)
                else:
                    for bit in (0, 1):
                        np_path = {
                            "pm": path["pm"] + self._path_metric_penalty(llr_bit, bit),
                            "L": L.copy(),
                            "B": B.copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        np_path["B"][l, n] = bit
                        np_path["u_hat"][l] = bit
                        self._update_bits(np_path["B"], l, n)
                        new_paths.append(np_path)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info = p["u_hat"][self.frozen_bits == 0]
                if crc_check(info, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]

    @staticmethod
    def _update_bits(B, l, n):
        if l < B.shape[0] // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                    B[j, s - 1] = B[j, s]
