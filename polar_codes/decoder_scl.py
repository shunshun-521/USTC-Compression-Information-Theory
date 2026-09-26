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
    _h_decision,
)


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特末尾。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _pm_update(pm, llr_bit, u_bit):
    """路径度量更新：与 LLR 不一致时加 |LLR|。"""
    preferred = 0 if llr_bit >= 0 else 1
    if int(u_bit) != preferred:
        pm += abs(llr_bit)
    return pm


class _Path:
    __slots__ = ("P", "C", "pm", "u_hat")

    def __init__(self, n, N, llr_ch):
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=int)
        self.P[n, :] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)

    def copy(self):
        n = self.P.shape[0] - 1
        N = self.P.shape[1]
        new = _Path(n, N, self.P[-1, :])
        new.P = self.P.copy()
        new.C = self.C.copy()
        new.pm = self.pm
        new.u_hat = self.u_hat.copy()
        return new

    def _prepare_llr(self, phi, n, N):
        l = 0
        while l < n and ((phi >> l) & 1):
            l += 1
        for li in range(l, n):
            stride = 1 << (n - 1 - li)
            if (phi % (2 * stride)) < stride:
                self.P[li, phi] = f_operation(
                    self.P[li + 1, phi], self.P[li + 1, phi + stride]
                )
        for li in range(l):
            stride = 1 << li
            self.P[li, phi] = g_operation(
                self.P[li + 1, phi - stride],
                self.P[li + 1, phi],
                self.C[li, phi - stride],
            )
        return self.P[0, phi]

    def _propagate_bit(self, phi, n):
        self.C[0, phi] = self.u_hat[phi]
        l = 0
        while l < n and ((phi >> l) & 1):
            self.C[l + 1, phi] = self.C[l, phi] ^ self.C[l, phi - (1 << l)]
            l += 1


class SCLDecoder:
    """SCL 译码器（路径复制 + PM 裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0
        N = self.N
        n = self.n
        paths = [_Path(n, N, llr_ch)]

        for phi in range(N):
            new_paths = []
            for path in paths:
                llr_bit = path._prepare_llr(phi, n, N)
                if self.frozen_bits[phi]:
                    p = path.copy()
                    p.u_hat[phi] = 0
                    p.pm = _pm_update(p.pm, llr_bit, 0)
                    p._propagate_bit(phi, n)
                    new_paths.append(p)
                else:
                    for bit in (0, 1):
                        p = path.copy()
                        p.u_hat[phi] = bit
                        p.pm = _pm_update(p.pm, llr_bit, bit)
                        p._propagate_bit(phi, n)
                        new_paths.append(p)
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


def scl_equivalent_to_sc(N, frozen_bits, llr):
    """L=1 的 SCL 应与 SC 一致（用于单元测试）。"""
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0).decode(llr)
    u_sc = sc_decode(llr, frozen_bits)
    return np.array_equal(u_scl, u_sc)
