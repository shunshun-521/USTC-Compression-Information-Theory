"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from encoder import bit_reversed_index
from decoder_sc import f_operation, g_operation, _hard_decision, _active_llr_level, _active_bit_level


# CRC-8: 0x07 (x^8+x^2+x+1), CRC-16: 0x8005
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    msb = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & msb:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径共享 LLR/比特数组，分裂时复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _path_metric_penalty(self, llr, bit):
        """与 LLR 符号一致不惩罚，否则加 |LLR|"""
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def _clone_path(self, path):
        new_p = _Path(self.N, self.n)
        new_p.L = path.L.copy()
        new_p.B = path.B.copy()
        new_p.pm = path.pm
        new_p.u_hat = path.u_hat.copy()
        return new_p

    def _update_llrs(self, path, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        np.array([path.L[j, s]]), np.array([path.L[j + branch_size, s]])
                    )[0]
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )  # (L_top, L_btm, u)

    def _update_bits(self, path, l):
        if l < self.N / 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            for j in range(l, -1, -block_size):
                if j % block_size >= block_size // 2:
                    path.B[j - block_size // 2, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - block_size // 2, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """
        SCL 译码。

        返回：
            u_hat: 最优路径估计
            pm: 最优路径度量
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        decode_order = [bit_reversed_index(i, self.n) for i in range(self.N)]

        for l in decode_order:
            for p in paths:
                self._update_llrs(p, l)

            new_paths = []
            for p in paths:
                llr_bit = p.L[l, self.n]
                if l in self.frozen_set:
                    pen = self._path_metric_penalty(llr_bit, 0)
                    p.pm += pen
                    p.u_hat[l] = 0
                    p.B[l, self.n] = 0
                    self._update_bits(p, l)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        cp = self._clone_path(p)
                        cp.pm += self._path_metric_penalty(llr_bit, bit)
                        cp.u_hat[l] = bit
                        cp.B[l, self.n] = bit
                        self._update_bits(cp, l)
                        new_paths.append(cp)

            new_paths.sort(key=lambda x: x.pm)
            paths = new_paths[: self.list_size]

        # 选择最优路径
        best = None
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best is None or p.pm < best.pm:
                        best = p
        if best is None:
            best = min(paths, key=lambda x: x.pm)

        return best.u_hat.copy(), best.pm
