"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math

from decoder_sc import f_operation, g_operation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
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
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


def _penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return float(abs(llr)) if int(bit) != hard else 0.0


class SCLDecoder:
    """SCL 译码器（极化树递归 + 路径裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = int(N)
        self.m = int(math.log2(self.N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        if self.frozen.dtype != np.bool_:
            self.frozen = self.frozen.astype(bool)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.info_indices = np.where(~self.frozen)[0]

        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.uint8)]

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append(
                    (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
                )
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [
            g_operation(a[p], b[p], beta_upper[p].astype(np.int64))
            for p in range(len(beta_upper))
        ]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.uint8)]
        _, _ = self._node([llr_ch], 0, self.N)

        paths = list(zip(self.metrics, self.decisions, strict=True))
        paths.sort(key=lambda x: x[0])

        best_crc = None
        if self.crc_length > 0:
            for pm, dec in paths:
                info_bits = dec[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or pm < best_crc[0]:
                        best_crc = (pm, dec)

        if best_crc is not None:
            pm, dec = best_crc
        else:
            pm, dec = paths[0]
        return dec.astype(int), float(pm)
