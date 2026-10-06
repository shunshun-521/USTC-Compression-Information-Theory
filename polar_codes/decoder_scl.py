"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly = _CRC8_POLY
    elif crc_length == 16:
        poly = _CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)


class SCLDecoder:
    """SCL 译码器（路径复制在分裂时执行，规模适中）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [
            int(format(i, f"0{self.n}b")[::-1], 2) for i in range(N)
        ]

    def _path_llr(self, path, l):
        _update_llrs(path.L, path.B, l, self.n)
        return path.L[l, self.n]

    def _pm_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [_Path(N, n)]
        paths[0].L[:, 0] = llr_ch

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                llr = self._path_llr(path, l)
                if self.frozen_bits[l]:
                    pm = self._pm_update(path.pm, llr, 0)
                    child = _Path(N, n)
                    child.L[:] = path.L
                    child.B[:] = path.B
                    child.pm = pm
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[l] = 0
                    child.B[l, n] = 0
                    _update_bits(child.B, l, n, N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        pm = self._pm_update(path.pm, llr, u)
                        child = _Path(N, n)
                        child.L[:] = path.L
                        child.B[:] = path.B
                        child.pm = pm
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[l] = u
                        child.B[l, n] = u
                        _update_bits(child.B, l, n, N)
                        new_paths.append(child)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
