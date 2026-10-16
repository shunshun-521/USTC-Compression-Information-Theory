"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _pm_penalty

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_process(info_bits, poly, crc_length)
    crc_bits = np.array([(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（树递归 + 路径裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.N = N
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]
        self.metrics = [0.0]
        self.decisions = [np.zeros(N, dtype=int)]

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((self.metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit))
        candidates.sort(key=lambda x: x[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.int8))
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

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=int)]
        self._node([llr], 0, self.N)

        paths = list(zip(self.metrics, self.decisions, strict=True))
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            for pm, u_hat in paths:
                info_bits = u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return u_hat.copy(), pm

        pm, u_hat = paths[0]
        return u_hat.copy(), pm
