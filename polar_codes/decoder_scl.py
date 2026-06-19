"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _li,
    _prepare_llr,
    path_metric_update,
    precompute_sc_indices,
)


CRC8_POLY_BITS = np.array([1, 0, 0, 0, 0, 0, 1, 1, 1], dtype=np.int8)  # x^8+x^2+x+1
CRC16_POLY_BITS = np.array(
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1], dtype=np.int8
)


def _crc_polynomial_bits(crc_length):
    return CRC8_POLY_BITS if crc_length == 8 else CRC16_POLY_BITS


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（系统多项式按位异或）。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_polynomial_bits(crc_length)
    plen = len(poly)
    msg = np.zeros(len(info_bits) + plen - 1, dtype=np.int8)
    msg[: len(info_bits)] = info_bits
    for i in range(len(msg) - plen + 1):
        if msg[i] == 1:
            msg[i : i + plen] ^= poly
    return np.concatenate([info_bits, msg[-(plen - 1) :]])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    poly = _crc_polynomial_bits(crc_length)
    plen = len(poly)
    msg = bits.copy()
    for i in range(len(msg) - plen + 1):
        if msg[i] == 1:
            msg[i : i + plen] ^= poly
    return np.all(msg[-(plen - 1) :] == 0)


class _Path:
    __slots__ = ("pm", "llrs", "s", "u_hat")

    def __init__(self, n, N):
        self.pm = 0.0
        self.llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
        self.s = np.full((n + 1, N), -1, dtype=np.int8)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器（路径分裂时复制 llrs/s）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]
        precompute_sc_indices(N)

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)。"""
        N, n = self.N, self.n
        llr = _prepare_llr(llr_ch, N)

        paths = [_Path(n, N)]
        paths[0].llrs[n, :] = llr

        for phi in range(N):
            candidates = []
            for path in paths:
                if self.frozen_bits[phi]:
                    cur = _li(0, phi, path.llrs, path.s)
                    path.pm = path_metric_update(path.pm, cur, 0)
                    path.u_hat[phi] = 0
                    path.s[0, phi] = 0
                    path.llrs[0, phi] = np.inf
                    candidates.append((path.pm, path))
                else:
                    cur = _li(0, phi, path.llrs, path.s)
                    for u in (0, 1):
                        child = _Path(n, N)
                        child.llrs = path.llrs.copy()
                        child.s = path.s.copy()
                        child.u_hat = path.u_hat.copy()
                        child.pm = path_metric_update(path.pm, cur, u)
                        child.u_hat[phi] = u
                        child.s[0, phi] = u
                        child.llrs[0, phi] = cur
                        candidates.append((child.pm, child))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[: self.list_size]]

        paths.sort(key=lambda p: p.pm)
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_idx]
                if crc_check(info_bits, self.crc_length):
                    return path.u_hat.copy(), path.pm
        return paths[0].u_hat.copy(), paths[0].pm
