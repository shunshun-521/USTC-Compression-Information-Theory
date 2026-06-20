"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversed
from decoder_sc import (
    active_bit_level,
    active_llr_level,
    f_operation,
    g_operation,
    sc_decode,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) & ((1 << crc_length) - 1)) ^ poly
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array([(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _path_metric_update(pm, llr, u):
    """路径度量更新：与 LLR 硬判决不一致时加 |LLR|。"""
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """
    SCL 译码器（含 Lazy Copy 优化）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：(u_hat, pm)
        """
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.zeros((N, n + 1), dtype=np.float64),
            "B": np.zeros((N, n + 1), dtype=int),
            "pm": 0.0,
        }]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]

                if l in self.frozen_set:
                    pm = _path_metric_update(path["pm"], llr, 0)
                    path["B"][l, n] = 0
                    self._update_bits(path["B"], l, n, N)
                    candidates.append({
                        "L": path["L"],
                        "B": path["B"],
                        "pm": pm,
                    })
                else:
                    for u in (0, 1):
                        L_copy = path["L"].copy()
                        B_copy = path["B"].copy()
                        pm = _path_metric_update(path["pm"], llr, u)
                        B_copy[l, n] = u
                        self._update_bits(B_copy, l, n, N)
                        candidates.append({
                            "L": L_copy,
                            "B": B_copy,
                            "pm": pm,
                        })

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        best = paths[0]
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["B"][:, n][self.info_indices], self.crc_length)]
            if valid:
                best = min(valid, key=lambda p: p["pm"])

        return best["B"][:, n].astype(int), best["pm"]

    @staticmethod
    def _update_llrs(L, B, l, n):
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, len(L), block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

    @staticmethod
    def _update_bits(B, l, n, N):
        if l < N // 2:
            return
        for s in range(n, n - active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]
