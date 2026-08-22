"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation, g_operation, _bit_reversed, _active_llr_level, _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]       # x^8 + x^2 + x + 1
_CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]  # CRC-16-IBM


def _gf2_crc_remainder(msg, poly):
    """GF(2) 多项式长除法求 CRC 余数"""
    msg = [int(b) for b in msg]
    poly = [int(b) for b in poly]
    n = len(poly) - 1
    for i in range(len(msg) - n):
        if msg[i]:
            for j in range(len(poly)):
                msg[i + j] ^= poly[j]
    return msg[-n:]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _gf2_crc_remainder(padded, poly)
    return np.concatenate([info_bits, np.array(rem, dtype=np.int8)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _gf2_crc_remainder(bits, poly)
    return all(r == 0 for r in rem)


# ==================== SCL 译码器 ====================

class _SCLPath:
    __slots__ = ('pm', 'L', 'B')

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时浅复制 L/B 数组）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new = _SCLPath(self.N, self.n, np.zeros(self.N))
        new.pm = path.pm
        new.L = path.L.copy()
        new.B = path.B.copy()
        return new

    def _update_llrs(self, path, l):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_bit = int(path.B[j - branch_size, s + 1])
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s], path.L[j, s], top_bit
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCLPath(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    path.pm += self._pm_penalty(llr, 0)
                    path.B[l, self.n] = 0
                    self._update_bits(path, l)
                    candidates.append(path)
                else:
                    for u in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, u)
                        new_path.B[l, self.n] = u
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        crc_paths = []
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.B[:, self.n].astype(int)[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(p)

        best = min(crc_paths or paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm
