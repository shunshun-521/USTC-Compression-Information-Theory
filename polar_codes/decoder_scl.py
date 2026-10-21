"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import f_operation, g_operation


_CRC8_CONFIG = (8, 0x07, 0x00, 0x00, False, False, 0x00)
_CRC16_CONFIG = (16, 0x8005, 0x0000, 0x0000, False, False, 0x0000)


def _reflect_bits(value, n_bits):
    result = 0
    for _ in range(n_bits):
        result = (result << 1) | (value & 1)
        value >>= 1
    return result


def _crc_bitwise(data: bytes, config) -> int:
    width, poly, init, xorout, refin, refout = config[:6]
    mask = (1 << width) - 1
    top_bit = 1 << (width - 1)
    crc = init & mask
    for raw_byte in data:
        in_byte = _reflect_bits(raw_byte, 8) if refin else raw_byte
        crc ^= in_byte << (width - 8)
        for _ in range(8):
            if crc & top_bit:
                crc = ((crc << 1) ^ poly) & mask
            else:
                crc = (crc << 1) & mask
    if refout:
        crc = _reflect_bits(crc, width)
    return crc ^ xorout


def _crc_config(crc_length):
    if crc_length == 8:
        return _CRC8_CONFIG
    if crc_length == 16:
        return _CRC16_CONFIG
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    config = _crc_config(crc_length)
    packed = np.packbits(info_bits.astype(np.uint8))
    value = _crc_bitwise(packed.tobytes(), config)
    crc_bits = np.array(
        [(value >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    config = _crc_config(crc_length)
    packed = np.packbits(bits.astype(np.uint8))
    return _crc_bitwise(packed.tobytes(), config) == 0


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器（精确路径度量，列表裁剪）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        if info_indices is None:
            self.info_indices = np.where(~self.frozen_bits)[0]
        else:
            self.info_indices = np.asarray(info_indices, dtype=int)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.uint8)]
        self._node([llr_ch], metrics, decisions, 0, self.N)
        paths = [(metrics[i], decisions[i].astype(int)) for i in range(len(metrics))]
        paths.sort(key=lambda p: p[0])
        if self.crc_length > 0:
            valid = [
                p
                for p in paths
                if crc_check(p[1][self.info_indices], self.crc_length)
            ]
            chosen = valid[0] if valid else paths[0]
        else:
            chosen = paths[0]
        return chosen[1].astype(int), chosen[0]

    def _leaf(self, llrs, metrics, decisions, index):
        if self.frozen_bits[index]:
            new_metrics = []
            new_decisions = []
            for path, llr in enumerate(llrs):
                m = metrics[path] + _penalty(float(llr[0]), 0)
                d = decisions[path].copy()
                d[index] = 0
                new_metrics.append(m)
                new_decisions.append(d)
            metrics[:] = new_metrics
            decisions[:] = new_decisions
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append(
                    (metrics[path] + _penalty(float(llr[0]), bit), path, bit)
                )
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            d = decisions[path].copy()
            d[index] = bit
            new_decisions.append(d)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)
        metrics[:] = new_metrics
        decisions[:] = new_decisions
        return betas, parent_map

    def _node(self, llrs, metrics, decisions, base, length):
        if length == 1:
            return self._leaf(llrs, metrics, decisions, base)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, metrics, decisions, base, half)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, metrics, decisions, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
