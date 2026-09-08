"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    bit_reversed, upper_llr, lower_llr,
    active_llr_level, active_bit_level,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_divide(info_bits, poly, crc_length):
    bits = list(info_bits) + [0] * crc_length
    poly_bits = [(poly >> i) & 1 for i in range(crc_length, -1, -1)]
    for i in range(len(info_bits)):
        if bits[i] == 1:
            for j, p in enumerate(poly_bits):
                bits[i + j] ^= p
    return np.array(bits[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = _crc_divide(info_bits, poly, crc_length)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected = crc_encode(info, crc_length)
    return np.array_equal(bits, expected)


class Path:
    """单条译码路径"""

    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        p = Path(self.L.shape[0], int(math.log2(self.L.shape[0])), self.L[:, 0])
        p.L[:] = self.L
        p.B[:] = self.B
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SCD + Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~np.asarray(frozen_bits, dtype=bool))[0]

    def _update_llrs(self, path, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = lower_llr(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(
                        path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, u):
        u_hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == u_hard else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        paths = [Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_val = path.L[l, self.n]

                if l in self.frozen:
                    for u in (0,):
                        pm = path.pm + self._pm_penalty(llr_val, u)
                        candidates.append((pm, path, u))
                else:
                    for u in (0, 1):
                        pm = path.pm + self._pm_penalty(llr_val, u)
                        candidates.append((pm, path, u))

            candidates.sort(key=lambda x: x[0])
            candidates = candidates[: self.L]

            new_paths = []
            for pm, parent, u in candidates:
                child = parent.copy()
                child.pm = pm
                child.u_hat[l] = u
                child.B[l, self.n] = u
                self._update_bits(child, l)
                new_paths.append(child)
            paths = new_paths

        best_any = min(paths, key=lambda p: p.pm)

        if self.crc_length > 0:
            crc_pass = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    crc_pass.append(p)
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)
                return best.u_hat, best.pm

        return best_any.u_hat, best_any.pm
