"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
)
from encoder import bit_reversal_permutation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.rev = bit_reversal_permutation(N)

    def _path_metric_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, paths, phi):
        l = _bit_reversed_index(phi, self.n)
        start = self.n - _active_llr_level(l, self.n)
        for path in paths:
            L, B = path["L"], path["B"]
            for s in range(start, self.n):
                block_size = 2 ** (s + 1)
                branch_size = block_size // 2
                for j in range(l, self.N, block_size):
                    if j % block_size < branch_size:
                        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                    else:
                        top_bit = B[j - branch_size, s + 1]
                        L[j, s + 1] = g_operation(
                            L[j - branch_size, s], L[j, s], top_bit
                        )

    def _update_bits(self, paths, phi):
        l = _bit_reversed_index(phi, self.n)
        if l < self.N // 2:
            return
        end = self.n - _active_bit_level(l, self.n)
        for path in paths:
            B = path["B"]
            for s in range(self.n, end, -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr_br = llr_ch[self.rev]

        paths = [
            {
                "pm": 0.0,
                "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
                "B": np.zeros((self.N, self.n + 1), dtype=int),
                "u_hat": np.zeros(self.N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr_br

        for phi in range(self.N):
            self._update_llrs(paths, phi)
            l = _bit_reversed_index(phi, self.n)
            llr = paths[0]["L"][l, self.n] if len(paths) == 1 else None

            new_paths = []
            for path in paths:
                cur_llr = path["L"][l, self.n]
                if self.frozen_bits[l]:
                    penalty = self._path_metric_penalty(cur_llr, 0)
                    path["pm"] += penalty
                    path["B"][l, self.n] = 0
                    path["u_hat"][l] = 0
                    new_paths.append(path)
                else:
                    for u_cand in (0, 1):
                        p = {
                            "pm": path["pm"] + self._path_metric_penalty(cur_llr, u_cand),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        p["B"][l, self.n] = u_cand
                        p["u_hat"][l] = u_cand
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.list_size]
            self._update_bits(paths, phi)

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            best = min(crc_pass, key=lambda p: p["pm"]) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best["u_hat"].copy(), best["pm"]
