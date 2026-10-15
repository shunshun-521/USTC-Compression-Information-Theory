"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _bits_to_bytes(bits):
    bits = np.asarray(bits, dtype=np.uint8).ravel()
    nbytes = (len(bits) + 7) // 8
    out = bytearray(nbytes)
    for i, b in enumerate(bits):
        if b:
            out[i // 8] |= 1 << (7 - (i % 8))
    return bytes(out)


def _crc_byte(crc, byte, poly):
    crc ^= byte
    for _ in range(8):
        if crc & 0x80:
            crc = ((crc << 1) ^ poly) & 0xFF
        else:
            crc = (crc << 1) & 0xFF
    return crc


def _crc_bytes(data, poly):
    crc = 0
    for b in data:
        crc = _crc_byte(crc, b, poly)
    return crc


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    data = _bits_to_bytes(info_bits)
    crc_val = _crc_bytes(data, poly)
    if crc_length == 16:
        hi = _crc_byte(0, (crc_val >> 8) & 0xFF, poly)
        lo = _crc_byte(hi, crc_val & 0xFF, poly)
        crc_val = ((hi << 8) | lo) & 0xFFFF
    crc_bits = np.array([(crc_val >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    data = _bits_to_bytes(bits)
    return _crc_bytes(data, poly) == 0


def _pm_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(float(llr))


class SCLDecoder:
    """SCL 译码器（路径列表 + 度量；L=1 时等价 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, int(list_size))
        self.block_length = int(N)
        self.crc_length = int(crc_length)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.int8)]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.block_length, dtype=np.int8)]
        _, _ = self._node([llr], 0, self.block_length)

        paths = list(zip(self.metrics, self.decisions, strict=True))
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            valid = [(m, d) for m, d in paths if crc_check(d, self.crc_length)]
            if valid:
                paths = valid

        pm, u_hat = paths[0]
        return np.asarray(u_hat, dtype=int), float(pm)

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.array([0], dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((self.metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit))
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)
        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation(l[:half], l[half:]) for l in llrs]
        beta_up, map_up = self._node(upper, base, half)

        lower = []
        for p, beta in enumerate(beta_up):
            parent = map_up[p]
            l = llrs[parent]
            lower.append(g_operation(l[:half], l[half:], beta))

        beta_low, map_low = self._node(lower, base + half, half)
        beta_up = [beta_up[map_low[p]] for p in range(len(map_low))]
        betas = [
            np.concatenate([beta_up[p] ^ beta_low[p], beta_low[p]])
            for p in range(len(beta_low))
        ]
        parent_map = [map_up[map_low[p]] for p in range(len(map_low))]
        return betas, parent_map
