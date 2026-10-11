"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np
from numpy.typing import NDArray


def _f(a, b):
    return np.asarray(np.logaddexp(0.0, a + b) - np.logaddexp(a, b), dtype=np.float64)


def _g(a, b, u):
    return np.asarray(b + (1.0 - 2.0 * u.astype(np.float64)) * a, dtype=np.float64)


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8) % 2
    if crc_length == 8:
        poly, width = 0x07, 8
    elif crc_length == 16:
        poly, width = 0x8005, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    mask = (1 << width) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(width):
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8) % 2
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _SCLCore:
    def __init__(self, frozen, list_size):
        self.frozen = frozen
        self.list_size = list_size
        self.block_length = int(frozen.size)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]

    def decode(self, channel_llr):
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.uint8)]
        llr = np.asarray(channel_llr, dtype=np.float64)
        codewords, _ = self._node([llr], 0, self.block_length)
        return list(zip(self.metrics, self.decisions, codewords, strict=True))

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
    frozen = np.asarray(frozen, dtype=bool)
    return _SCLCore(frozen, list_size).decode(channel_llr)


class SCLDecoder:
    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.list_size = list_size
        self.crc_length = crc_length
        fb = np.asarray(frozen_bits)
        if fb.dtype == bool:
            self.frozen = fb
        else:
            self.frozen = fb.astype(int) != 0
        self.info_idx = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        paths = scl_decode(np.asarray(llr_ch, dtype=np.float64), self.frozen, self.list_size)
        paths = sorted(paths, key=lambda p: p[0])
        if self.crc_length > 0:
            for pm, u_hat, _ in paths:
                payload = u_hat[self.info_idx]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), pm
        pm, u_hat, _ = paths[0]
        return u_hat.astype(int), pm
