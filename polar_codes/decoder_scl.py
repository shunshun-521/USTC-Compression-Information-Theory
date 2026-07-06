"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)
from encoder import bit_reversed


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 7
            for _ in range(8):
                if reg & 0x80:
                    reg = ((reg << 1) ^ CRC8_POLY) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
        crc_bits = np.array([(reg >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    else:
        reg = 0
        for bit in info_bits:
            reg ^= int(bit) << 15
            for _ in range(8):
                if reg & 0x8000:
                    reg = ((reg << 1) ^ CRC16_POLY) & 0xFFFF
                else:
                    reg = (reg << 1) & 0xFFFF
        crc_bits = np.array([(reg >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    return np.array_equal(bits, crc_encode(bits[:-crc_length], crc_length))


class _Path:
    __slots__ = ("pm", "L", "B", "active")

    def __init__(self, n, N, llr_ch):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 L/B）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.n, self.N, llr_ch)]

        for phi in range(self.N):
            l = bit_reversed(phi, self.n)
            new_paths = []

            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    pm = path.pm + (0.0 if llr >= 0.0 else abs(llr))
                    child = _Path(self.n, self.N, llr_ch)
                    child.pm = pm
                    child.L = path.L.copy()
                    child.B = path.B.copy()
                    child.B[l, self.n] = 0
                    _update_bits(child.B, l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        pm = path.pm
                        hard = 0 if llr >= 0.0 else 1
                        if bit != hard:
                            pm += abs(llr)
                        child = _Path(self.n, self.N, llr_ch)
                        child.pm = pm
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.B[l, self.n] = bit
                        _update_bits(child.B, l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        best_u = None
        best_pm = float("inf")

        for path in paths:
            u_hat = path.B[:, self.n].astype(np.int8)
            if self.crc_length > 0:
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length) and path.pm < best_pm:
                    best_pm = path.pm
                    best_u = u_hat
            elif path.pm < best_pm:
                best_pm = path.pm
                best_u = u_hat

        if best_u is None:
            best_u = paths[0].B[:, self.n].astype(np.int8)
            best_pm = paths[0].pm

        return best_u, best_pm
