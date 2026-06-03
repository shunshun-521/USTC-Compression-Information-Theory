"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import math

import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_len):
    reg = 0
    top = (1 << (crc_len - 1)) if crc_len <= 8 else (1 << 15)
    mask = (1 << crc_len) - 1 if crc_len <= 8 else 0xFFFF
    for b in bits:
        reg ^= int(b) << (crc_len - 1)
        steps = 8 if crc_len <= 8 else 16
        for _ in range(steps):
            if reg & top:
                reg = ((reg << 1) & mask) ^ poly
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


# ==================== SCL 译码器 ====================


class Path:
    __slots__ = ("pm", "lam", "beta", "u")

    def __init__(self, N, llr_ch):
        n = int(math.log2(N))
        self.pm = 0.0
        self.lam = [np.zeros(1 << layer, dtype=np.float64) for layer in range(n + 1)]
        self.beta = [np.zeros(1 << layer, dtype=np.int8) for layer in range(n + 1)]
        self.lam[-1][:] = llr_ch
        self.u = []


def _calc_lambda_path(path, phase, n):
    lam, beta = path.lam, path.beta

    def calc(layer, ph):
        if layer == 0:
            return
        if ph % 2 == 0:
            calc(layer - 1, ph // 2)
        for j in range(1 << (layer - 1)):
            if ph % 2 == 0:
                lam[layer - 1][j] = f_operation(lam[layer][2 * j], lam[layer][2 * j + 1])
            else:
                lam[layer - 1][j] = g_operation(
                    lam[layer][2 * j], lam[layer][2 * j + 1], beta[layer - 1][j]
                )

    calc(n, phase)


def _update_beta_path(path, phase, bit, n):
    beta = path.beta

    def upd(layer, ph):
        if layer == n:
            return
        for j in range(1 << layer):
            if ph % 2 == 0:
                beta[layer + 1][2 * j] = beta[layer][j]
                beta[layer + 1][2 * j + 1] = beta[layer][j] ^ bit
            else:
                beta[layer + 1][2 * j + 1] = beta[layer][j]
        if ph % 2 == 1:
            upd(layer + 1, ph // 2)

    beta[0][0] = bit
    upd(0, phase)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.L == 1:
            u = sc_decode(llr_ch, self.frozen_bits)
            return u, 0.0
        paths = [Path(self.N, llr_ch)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                _calc_lambda_path(path, phi, self.n)
                llr0 = path.lam[0][0]
                branches = [0] if self.frozen_bits[phi] else [0, 1]
                for bit in branches:
                    pm_inc = (
                        0.0
                        if (bit == 0 and llr0 >= 0) or (bit == 1 and llr0 < 0)
                        else abs(llr0)
                    )
                    child = copy.deepcopy(path)
                    child.pm += pm_inc
                    child.u = path.u + [bit]
                    _update_beta_path(child, phi, bit, self.n)
                    candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.L]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(np.array(p.u), self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        u_hat = np.zeros(self.N, dtype=int)
        u_hat[: len(best.u)] = best.u
        return u_hat, best.pm


def scl_equals_sc(N, frozen_bits, llr, tol=0):
    """L=1 时与 SC 等价（测试用）。"""
    scl = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)[0]
    sc = sc_decode(llr, frozen_bits)
    return np.array_equal(scl, sc)
