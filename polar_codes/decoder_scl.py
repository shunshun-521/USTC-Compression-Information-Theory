"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    sc_decode,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)


CRC8_POLY = [1, 0, 0, 0, 0, 0, 1, 1, 1]   # x^8 + x^2 + x + 1
CRC16_POLY = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # 0x8005


def _poly_div_crc(message_bits, generator_bits):
    g = list(generator_bits)
    r = len(g) - 1
    reg = list(message_bits) + [0] * r
    for i in range(len(message_bits)):
        if reg[i] == 1:
            for j in range(len(g)):
                reg[i + j] ^= g[j]
    return np.array(reg[-r:], dtype=int)


def _poly_div_check(bits, generator_bits):
    g = list(generator_bits)
    r = len(g) - 1
    reg = list(bits)
    for i in range(len(bits) - r):
        if reg[i] == 1:
            for j in range(len(g)):
                reg[i + j] ^= g[j]
    return all(x == 0 for x in reg[-r:])


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _poly_div_crc(info_bits.tolist(), poly)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _poly_div_check(bits.tolist(), poly)


class PathState:
    """单条译码路径（Lazy Copy）。"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n, llr_ch=None, parent=None):
        self.pm = 0.0
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=int)
            if llr_ch is not None:
                self.L[:, 0] = llr_ch
            self.u_hat = np.zeros(N, dtype=int)
        else:
            self.L = parent.L
            self.B = parent.B
            self.u_hat = parent.u_hat.copy()
            self.pm = parent.pm


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _pm_update(self, pm, llr, u):
        hard = 0 if llr >= 0 else 1
        return pm + (0.0 if u == hard else abs(llr))

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block >> 1
            for j in range(l, self.N, block):
                if j % block < half:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + half, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - half, s], path.L[j, s], path.B[j - half, s + 1]
                    )

    def _update_bits(self, path, l, u_bit):
        path.B[l, self.n] = u_bit
        if l >= self.N // 2:
            for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
                block = 1 << s
                half = block >> 1
                for j in range(l, -1, -block):
                    if j % block >= half:
                        path.B[j - half, s - 1] = path.B[j, s] ^ path.B[j - half, s]
                        path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        paths = [PathState(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]
                if self.frozen_bits[phi]:
                    candidates.append((self._pm_update(path.pm, llr, 0), path, 0))
                else:
                    for u in (0, 1):
                        candidates.append((self._pm_update(path.pm, llr, u), path, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.list_size]

            new_paths = []
            for pm, parent, u_bit in candidates:
                child = PathState(self.N, self.n, parent=parent)
                child.pm = pm
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.u_hat[phi] = u_bit
                self._update_bits(child, l, u_bit)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat, best.pm
