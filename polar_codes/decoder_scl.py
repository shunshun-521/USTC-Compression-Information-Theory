"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def _crc_run(bits, crc_length):
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    top = 1 << (crc_length - 1)
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """CRC-8 (0x07) 或 CRC-16 (0x8005)"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    reg = _crc_run(info_bits, crc_length)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    return _crc_run(bits, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（路径度量 + 列表裁剪 + 可选 CRC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        if self.frozen.dtype != bool:
            self.frozen = self.frozen != 0
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.info_positions = np.where(~self.frozen)[0]
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int32)]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int32)]
        _, _ = self._node([llr], 0, self.N)
        paths = sorted(
            zip(self.metrics, self.decisions, strict=True), key=lambda x: x[0]
        )

        if self.crc_length > 0:
            for pm, u_hat in paths:
                info = u_hat[self.info_positions]
                if crc_check(info, self.crc_length):
                    return u_hat, pm

        return paths[0][1], paths[0][0]

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int32) for _ in llrs], list(range(len(llrs)))

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
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.int32))
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

        lower = []
        for p in range(len(map_upper)):
            parent = map_upper[p]
            lower.append(
                g_operation(llrs[parent][:half], llrs[parent][half:], beta_upper[p])
            )
        beta_lower, map_lower = self._node(lower, base + half, half)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
