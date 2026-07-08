"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    bit_reversed,
    active_llr_level,
    active_bit_level,
)

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")

    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("pm", "B", "L", "active")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.active = True

    def copy(self):
        p = _Path.__new__(_Path)
        p.pm = self.pm
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.active = True
        return p


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
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

    def _update_bits(self, path, l, u_val):
        path.B[l, self.n] = u_val
        if l >= self.N // 2:
            for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                            path.B[j - branch_size, s]
                        )
                        path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr, u):
        u_from_llr = 0 if llr >= 0 else 1
        return 0.0 if u == u_from_llr else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n

        paths = [_Path(n, N)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = bit_reversed(i, n)
            new_paths = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, n]

                if l in self.frozen_set:
                    penalty = self._path_metric_penalty(llr, 0)
                    path.pm += penalty
                    self._update_bits(path, l, 0)
                    new_paths.append(path)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm += self._path_metric_penalty(llr, u)
                        self._update_bits(p, l, u)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_pass_paths = []
        if self.crc_length > 0:
            for p in paths:
                info_bits = p.B[:, n][self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    crc_pass_paths.append(p)

        if crc_pass_paths:
            best = min(crc_pass_paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.B[:, n].astype(int).copy(), best.pm
