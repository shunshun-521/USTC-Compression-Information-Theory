"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _active_bit_level, _active_llr_level
from encoder import bit_reversed


CRC8_POLY = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)
CRC16_POLY = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8
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
    r = crc_length
    padded = np.concatenate([info_bits, np.zeros(r, dtype=np.int8)])
    remainder = _crc_remainder(padded, poly)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    padded = np.concatenate([bits[:-crc_length], np.zeros(crc_length, dtype=np.int8)])
    remainder = _crc_remainder(padded, poly)
    return np.array_equal(remainder, bits[-crc_length:])


class PathState:
    __slots__ = ("pm", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]

    def _copy_path(self, path):
        new = PathState(self.N, self.n)
        new.pm = path.pm
        new.L = path.L.copy()
        new.B = path.B.copy()
        return new

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        paths = [PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    p = self._copy_path(path)
                    p.pm += self._pm_penalty(llr_val, 0)
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for u in (0, 1):
                        p = self._copy_path(path)
                        p.pm += self._pm_penalty(llr_val, u)
                        p.B[l, self.n] = u
                        self._update_bits(p, l)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.B[:, self.n][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p.pm)
        else:
            best = paths[0]

        return best.B[:, self.n].astype(int), best.pm
