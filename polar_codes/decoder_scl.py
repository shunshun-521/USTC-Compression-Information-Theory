"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import SCDecoder, _bit_reversed, _prepare_channel_llr

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        feedback = ((reg >> (crc_length - 1)) ^ int(bit)) & 1
        reg = ((reg << 1) & mask) ^ (poly if feedback else 0)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否满足 CRC。"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class _PathState:
    __slots__ = ("pm", "L", "B", "u_hat")

    def __init__(self, N, n):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.u_hat = np.zeros(N, dtype=np.int8)


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_indices = np.where(~np.asarray(frozen_bits, dtype=bool))[0]
        self._sc = SCDecoder(N, frozen_bits)

    @staticmethod
    def _branch_penalty(llr_val, u):
        return 0.0 if (u == 0 and llr_val >= 0) or (u == 1 and llr_val < 0) else abs(llr_val)

    def _advance_path(self, path, l):
        self._sc.L = path.L
        self._sc.B = path.B
        self._sc._update_llrs(l)
        llr_val = self._sc.L[l, self.n]
        if l in self.frozen:
            path.pm += self._branch_penalty(llr_val, 0)
            path.u_hat[l] = 0
            path.B[l, self.n] = 0
            self._sc._update_bits(l)
            path.L = self._sc.L.copy()
            path.B = self._sc.B.copy()
            return [path]
        children = []
        for u in (0, 1):
            child = _PathState(self.N, self.n)
            child.pm = path.pm + self._branch_penalty(llr_val, u)
            child.L = path.L.copy()
            child.B = path.B.copy()
            child.u_hat = path.u_hat.copy()
            child.u_hat[l] = u
            child.B[l, self.n] = u
            self._sc.L = child.L
            self._sc.B = child.B
            self._sc._update_bits(l)
            child.L = self._sc.L.copy()
            child.B = self._sc.B.copy()
            children.append(child)
        return children

    def decode(self, llr_ch):
        llr_ch = _prepare_channel_llr(llr_ch)
        paths = [_PathState(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch
        decode_order = [_bit_reversed(i, self.n) for i in range(self.N)]
        for l in decode_order:
            new_paths = []
            for path in paths:
                new_paths.extend(self._advance_path(path, l))
            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_indices], self.crc_length)
            ]
            chosen = min(valid, key=lambda p: p.pm) if valid else min(paths, key=lambda p: p.pm)
        else:
            chosen = min(paths, key=lambda p: p.pm)
        return chosen.u_hat.astype(int), chosen.pm
