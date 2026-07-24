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
    _prepare_channel_llrs,
    f_operation,
    g_operation,
    sc_decode,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_metric_update(self, pm, llr, u):
        u_hard = 0 if llr >= 0 else 1
        if u != u_hard:
            pm += abs(llr)
        return pm

    def _update_llrs(self, path, l):
        L, B = path["L"], path["B"]
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 2 ** (s + 1)
            branch = block // 2
            for j in range(l, self.N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )

    def _update_bits(self, path, l):
        B = path["B"]
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 2 ** s
            branch = block // 2
            for j in range(l, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = (B[j, s] + B[j - branch, s]) % 2
                    B[j, s - 1] = B[j, s]

    def _new_path(self, llr):
        return {
            "pm": 0.0,
            "u_hat": np.zeros(self.N, dtype=int),
            "L": np.zeros((self.N, self.n + 1), dtype=np.float64),
            "B": np.zeros((self.N, self.n + 1), dtype=np.int32),
            "L_init": llr,
        }

    def _clone_path(self, path):
        p = {
            "pm": path["pm"],
            "u_hat": path["u_hat"].copy(),
            "L": path["L"].copy(),
            "B": path["B"].copy(),
            "L_init": path["L_init"],
        }
        return p

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        if self.list_size == 1:
            return sc_decode(llr_ch, self.frozen_bits), 0.0

        llr = _prepare_channel_llrs(llr_ch, self.N)
        paths = [self._new_path(llr)]
        paths[0]["L"][:, 0] = llr

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr_bit = path["L"][l, self.n]

                if self.frozen_bits[l]:
                    p = self._clone_path(path)
                    p["pm"] = self._path_metric_update(p["pm"], llr_bit, 0)
                    p["u_hat"][l] = 0
                    p["B"][l, self.n] = 0
                    self._update_bits(p, l)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = self._clone_path(path)
                        p["pm"] = self._path_metric_update(p["pm"], llr_bit, u)
                        p["u_hat"][l] = u
                        p["B"][l, self.n] = u
                        self._update_bits(p, l)
                        candidates.append(p)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        crc_pass = []
        for p in paths:
            if self.crc_length > 0:
                info_pos = np.where(self.frozen_bits == 0)[0]
                if crc_check(p["u_hat"][info_pos], self.crc_length):
                    crc_pass.append(p)
            else:
                crc_pass.append(p)

        best = min(crc_pass or paths, key=lambda p: p["pm"])
        return best["u_hat"].copy(), best["pm"]
