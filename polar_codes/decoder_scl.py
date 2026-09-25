"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import _active_bit_level, _active_llr_level, _bit_reversed_index, _lower_llr, _upper_llr
from encoder import bit_reversal_permutation


CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]          # x^8 + x^2 + x + 1 (0x07)
CRC16_POLY_BITS = [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # 0x8005


def _crc_poly_bits(crc_length):
    return CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS


def _crc_mod2_remainder(bits, poly_bits):
    """模 2 长除法求 CRC 余式。"""
    reg = [int(b) for b in bits]
    n = len(poly_bits)
    for i in range(len(reg) - n + 1):
        if reg[i] == 1:
            for j in range(n):
                reg[i + j] ^= poly_bits[j]
    return np.array(reg[-(n - 1):], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly_bits = _crc_poly_bits(crc_length)
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    crc_bits = _crc_mod2_remainder(msg, poly_bits)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly_bits = _crc_poly_bits(crc_length)
    remainder = _crc_mod2_remainder(bits, poly_bits)
    return np.all(remainder == 0)


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        llr = llr_ch[self.br]
        paths = [_Path(self.N, self.n, llr)]

        for phi in range(self.N):
            l = _bit_reversed_index(phi, self.n)
            new_paths = []

            for path in paths:
                for s in range(self.n - _active_llr_level(l, self.n), self.n):
                    block_size = 2 ** (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, self.N, block_size):
                        if j % block_size < branch_size:
                            path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                        else:
                            path.L[j, s + 1] = _lower_llr(
                                path.L[j, s],
                                path.L[j - branch_size, s],
                                int(path.B[j - branch_size, s + 1]),
                            )

                cur_llr = path.L[l, self.n]

                if l in self.frozen_set:
                    p = _Path(self.N, self.n, llr)
                    p.L = path.L.copy()
                    p.B = path.B.copy()
                    p.pm = path.pm + self._path_metric_penalty(cur_llr, 0)
                    p.u_hat = path.u_hat.copy()
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = _Path(self.N, self.n, llr)
                        p.L = path.L.copy()
                        p.B = path.B.copy()
                        p.pm = path.pm + self._path_metric_penalty(cur_llr, bit)
                        p.u_hat = path.u_hat.copy()
                        p.u_hat[l] = bit
                        p.B[l, self.n] = bit
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            for p in paths:
                if crc_check(p.u_hat, self.crc_length):
                    crc_paths.append(p)

        best = min(crc_paths or paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]
