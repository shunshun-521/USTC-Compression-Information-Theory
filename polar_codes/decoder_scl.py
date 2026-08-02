"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _penalty


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits[:-crc_length], poly, crc_length)
    actual = 0
    for b in bits[-crc_length:]:
        actual = (actual << 1) | int(b)
    return remainder == actual


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        paths = self._scl_decode(np.asarray(llr_ch, dtype=np.float64))
        paths = sorted(paths, key=lambda p: p[0])
        best = self._select_best_path(paths)
        return best[1].copy(), best[0]

    def _scl_decode(self, channel_llr):
        decoder = _SCLCore(self.frozen_bits, self.list_size)
        return decoder.decode(channel_llr)

    def _select_best_path(self, paths):
        if self.crc_length > 0:
            info_indices = np.where(~self.frozen_bits)[0]
            passed = [
                p for p in paths
                if crc_check(p[1][info_indices], self.crc_length)
            ]
            if passed:
                return min(passed, key=lambda p: p[0])
        return paths[0]


class _SCLCore:
    """SCL 译码核心。"""

    def __init__(self, frozen, list_size):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.list_size = list_size
        self.block_length = int(frozen.size)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=int)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=int)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        _, _ = self._node([llr], 0, self.block_length)
        return list(zip(self.metrics, self.decisions, strict=True))

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[:self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=int))
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
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
