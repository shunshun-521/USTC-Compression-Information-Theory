"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    prepare_channel_llr,
    precompute_sc_indices,
    _bit_rev,
    _active_llr_level,
    _active_bit_level,
    _update_llrs,
    _update_bits,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return _CRC8_POLY
    if crc_length == 16:
        return _CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def _crc_step(reg, bit, crc_length, poly):
    mask = (1 << crc_length) - 1
    fb = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
    reg = (reg << 1) & mask
    if fb:
        reg ^= poly
    return reg, fb


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg, _ = _crc_step(reg, bit, crc_length, poly)
    out = info_bits.tolist()
    for _ in range(crc_length):
        reg, fb = _crc_step(reg, 0, crc_length, poly)
        out.append(fb)
    return np.array(out, dtype=np.int8)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in np.asarray(bits, dtype=np.int8):
        reg, _ = _crc_step(reg, bit, crc_length, poly)
    return reg == 0


def _path_metric_update(pm, llr, u):
    expected = 0 if llr >= 0 else 1
    if u != expected:
        pm += abs(llr)
    return pm


class _Path:
    __slots__ = ("L", "B", "pm", "u_hat")

    def __init__(self, N, n, llr):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = _Path(self.L.shape[0], self.L.shape[1] - 1, np.zeros(self.L.shape[0]))
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy：路径分裂时复制 L/B）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        precompute_sc_indices(N)

    def decode(self, llr_ch):
        llr = prepare_channel_llr(llr_ch)
        n, N = self.n, self.N
        paths = [_Path(N, n, llr)]

        for i in range(N):
            l = _bit_rev(i, n)
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, n)
                llr0 = path.L[l, n]
                if l in self.frozen_set:
                    new_path = path.copy()
                    new_path.pm = _path_metric_update(new_path.pm, llr0, 0)
                    new_path.B[l, n] = 0
                    new_path.u_hat[l] = 0
                    _update_bits(new_path.B, l, n)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.copy()
                        new_path.pm = _path_metric_update(new_path.pm, llr0, u)
                        new_path.B[l, n] = u
                        new_path.u_hat[l] = u
                        _update_bits(new_path.B, l, n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            best = min(valid or paths, key=lambda p: p.pm)
        else:
            best = min(paths, key=lambda p: p.pm)

        return best.u_hat.copy(), best.pm
