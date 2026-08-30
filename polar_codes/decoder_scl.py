"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
基于 Permuted SCD (Vangala et al., 2014)
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import (
    f_operation, g_operation, active_llr_level, active_bit_level,
    _update_llrs, _update_bits,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int)
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(encoded, bits)


class _Path:
    """单条译码路径"""

    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        new_path = _Path(len(self.L), len(self.L[0]) - 1)
        new_path.pm = self.pm
        new_path.L = self.L.copy()
        new_path.B = self.B.copy()
        new_path.u_hat = self.u_hat.copy()
        return new_path


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        N, n = self.N, self.n
        llr = llr_ch.astype(np.float64)

        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr

        for phi in range(N):
            l = bit_reversed(phi, n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, n, N)
                cur_llr = path.L[l, n]

                if self.frozen_bits[l]:
                    new_p = path.copy()
                    new_p.pm += self._pm_penalty(cur_llr, 0)
                    new_p.u_hat[l] = 0
                    new_p.B[l, n] = 0
                    _update_bits(new_p.B, l, n, N)
                    new_paths.append(new_p)
                else:
                    for u in (0, 1):
                        new_p = path.copy()
                        new_p.pm += self._pm_penalty(cur_llr, u)
                        new_p.u_hat[l] = u
                        new_p.B[l, n] = u
                        _update_bits(new_p.B, l, n, N)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
