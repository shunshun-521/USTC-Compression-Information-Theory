"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _prepare_llr,
    _update_bits,
    _update_llrs,
    f_boxplus,
    g_boxplus,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_update(reg, bit, crc_length, poly):
    msb = 1 << (crc_length - 1)
    reg = ((reg << 1) | int(bit)) & ((1 << crc_length) - 1)
    if reg & msb:
        reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    for _ in range(crc_length):
        reg = _crc_update(reg, 0, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    return reg == 0


def _path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u == hard:
        return pm
    return pm + abs(llr)


class _Path:
    __slots__ = ("pm", "L", "B")

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def _clone_path(self, path):
        new_path = _Path(self.n, self.N)
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        return new_path

    def decode(self, llr_ch):
        llr = _prepare_llr(llr_ch)
        n = self.n
        paths = [_Path(n, self.N)]
        paths[0].L[:, 0] = llr

        for phi in range(self.N):
            l = _bit_reversed(phi, n)
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n, f_boxplus, g_boxplus)
                llr_val = path.L[l, n]
                if self.frozen_bits[l]:
                    child = self._clone_path(path)
                    child.pm = _path_metric_update(child.pm, llr_val, 0)
                    child.B[l, n] = 0
                    _update_bits(child.B, l, n)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = self._clone_path(path)
                        child.pm = _path_metric_update(child.pm, llr_val, u)
                        child.B[l, n] = u
                        _update_bits(child.B, l, n)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        candidates = []
        for path in paths:
            u_hat = path.B[:, n].astype(np.int8)
            candidates.append((path.pm, u_hat))

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in candidates if crc_check(u, self.crc_length)]
            pm, u_hat = min(valid if valid else candidates, key=lambda x: x[0])
        else:
            pm, u_hat = candidates[0]

        return u_hat.copy(), pm
