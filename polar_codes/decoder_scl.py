"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _f(a, b):
    return np.logaddexp(0.0, a + b) - np.logaddexp(a, b)


def _g(a, b, u):
    return b + (1.0 - 2.0 * u.astype(np.float64)) * a


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class _SCLCore:
    def __init__(self, frozen, list_size):
        self.frozen = np.asarray(frozen, dtype=bool)
        self.list_size = list_size
        self.N = frozen.size
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.uint8)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        codewords, _ = self._node([llr], 0, self.N)
        paths = list(zip(self.metrics, self.decisions, codewords, strict=True))
        return sorted(paths, key=lambda p: p[0])

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
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
        upper = [_f(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [_g(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map


def scl_decode(channel_llr, frozen, list_size):
    """返回 [(metric, u_hat, codeword), ...]"""
    return _SCLCore(frozen, list_size).decode(channel_llr)


class SCLDecoder:
    """SCL / CA-SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        if self.frozen.dtype != bool:
            self.frozen = self.frozen.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        paths = scl_decode(llr_ch, self.frozen, self.list_size)
        if self.crc_length > 0:
            for metric, u_hat, _ in paths:
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), metric
        metric, u_hat, _ = paths[0]
        return u_hat.astype(int), metric
