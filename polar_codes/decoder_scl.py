"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_bitwise(bits, crc_length):
    """逐比特 CRC 计算（MSB 优先）"""
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    top = 1 << (crc_length - 1)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_bitwise(padded, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确（与 crc_encode 自洽）"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    return np.array_equal(bits, crc_encode(payload, crc_length))


class PathState:
    """单条 SCL 路径"""

    __slots__ = ("L", "B", "pm", "active")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int_)
        self.pm = 0.0
        self.active = True

    def clone(self):
        new = PathState(self.L.shape[0], self.L.shape[1] - 1)
        new.L[:] = self.L
        new.B[:] = self.B
        new.pm = self.pm
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _metric_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L_size = self.list_size

        paths = [PathState(N, n)]
        paths[0].L[:, 0] = llr_ch

        for i in range(N):
            l = _bit_reversed_index(i, n)
            for path in paths:
                _update_llrs(path.L, path.B, l, n)

            if self.frozen_bits[l]:
                for path in paths:
                    path.B[l, n] = 0
                    path.pm += self._metric_penalty(path.L[l, n], 0)
                    _update_bits(path.B, l, n)
            else:
                candidates = []
                for path in paths:
                    for u in (0, 1):
                        child = path.clone()
                        child.B[l, n] = u
                        child.pm += self._metric_penalty(child.L[l, n], u)
                        _update_bits(child.B, l, n)
                        candidates.append(child)
                candidates.sort(key=lambda p: p.pm)
                paths = candidates[:L_size]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.B[:, n][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            best = valid[0] if valid else paths[0]
        else:
            best = paths[0]

        return best.B[:, n].astype(int), best.pm
