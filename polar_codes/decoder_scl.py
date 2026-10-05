"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversed
from decoder_sc import _update_llrs, _update_bits, f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_update(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    reg &= mask
    for _ in range(8):
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg = _crc_update(reg, b, poly, crc_length)
    for _ in range(crc_length):
        reg = _crc_update(reg, 0, poly, crc_length)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC（与 crc_encode 成对定义）。"""
    bits = np.asarray(bits, dtype=np.int8)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits, expected)


class _PathState:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, n, N):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（置换 SC 顺序）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def _pm_update(self, pm, llr, u):
        penalty = 0.0 if (u == 0 and llr >= 0) or (u == 1 and llr < 0) else abs(llr)
        return pm + penalty

    def _crc_pass(self, u_hat):
        if self.crc_length <= 0:
            return True
        if self.info_indices is not None:
            bits = u_hat[self.info_indices]
        else:
            bits = u_hat
        return crc_check(bits, self.crc_length)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n)
                llr = path.L[l, self.n]
                if self.frozen_bits[l]:
                    path.pm = self._pm_update(path.pm, llr, 0)
                    path.u_hat[l] = 0
                    path.B[l, self.n] = 0
                    _update_bits(path.B, l, self.n)
                    new_paths.append(path)
                else:
                    for u_bit in (0, 1):
                        child = _PathState(self.n, self.N)
                        child.L = path.L.copy()
                        child.B = path.B.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = self._pm_update(path.pm, llr, u_bit)
                        child.u_hat[l] = u_bit
                        child.B[l, self.n] = u_bit
                        _update_bits(child.B, l, self.n)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if self._crc_pass(p.u_hat)]
            if valid:
                best = min(valid, key=lambda p: p.pm)
            else:
                best = paths[0]
        else:
            best = paths[0]
        return best.u_hat.astype(int), best.pm
