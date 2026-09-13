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
    _lower_llr_exact,
    _upper_llr_exact,
)


_CRC8_GEN = [1, 0, 0, 0, 0, 0, 1, 1, 1]
_CRC16_GEN = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]


def _crc_remainder(bits, generator):
    msg = list(bits) + [0] * (len(generator) - 1)
    n = len(bits)
    for i in range(n):
        if msg[i]:
            for j in range(len(generator)):
                msg[i + j] ^= generator[j]
    return msg[n:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        generator = _CRC8_GEN
    elif crc_length == 16:
        generator = _CRC16_GEN
    else:
        raise ValueError("crc_length must be 8 or 16")
    crc_bits = np.array(_crc_remainder(info_bits, generator), dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if crc_length == 8:
        generator = _CRC8_GEN
    elif crc_length == 16:
        generator = _CRC16_GEN
    else:
        raise ValueError("crc_length must be 8 or 16")
    return all(r == 0 for r in _crc_remainder(bits, generator))


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _branch_penalty(llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _update_llrs(self, path, phi):
        l = _bit_reversed(phi, self.n)
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr_exact(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = _lower_llr_exact(
                        path.L[j, s], path.L[j - branch_size, s], top_bit
                    )
        return path.L[l, self.n]

    def _update_bits(self, path, phi, bit):
        l = _bit_reversed(phi, self.n)
        path.B[l, self.n] = bit
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回最优路径的 u_hat 和路径度量"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                llr = self._update_llrs(path, phi)

                if l in self.frozen_set:
                    path.pm += self._branch_penalty(llr, 0)
                    self._update_bits(path, phi, 0)
                    candidates.append(path)
                else:
                    for bit in (0, 1):
                        new_path = _Path(self.n, self.N)
                        new_path.pm = path.pm + self._branch_penalty(llr, bit)
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        self._update_bits(new_path, phi, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            chosen = valid[0] if valid else paths[0]
        else:
            chosen = paths[0]

        return chosen.B[:, self.n].astype(int), chosen.pm
