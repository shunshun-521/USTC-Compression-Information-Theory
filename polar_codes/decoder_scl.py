"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation, _pm_penalty


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg <<= 1
        reg |= int(b)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=np.int8).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(bits, poly, crc_length)
    return rem == 0


class SCLDecoder:
    """SCL 译码器（树形递归 + 路径裁剪）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.metrics = [0.0]
        self.decisions = [np.zeros(N, dtype=np.int8)]

    def _leaf(self, llrs, index):
        if self.frozen_bits[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append(
                    (self.metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit)
                )
        candidates.sort(key=lambda t: t[0])
        kept = candidates[: self.list_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
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

        lower = []
        for p, bu in enumerate(beta_upper):
            parent = map_upper[p]
            lower.append(
                g_operation(llrs[parent][:half], llrs[parent][half:], bu)
            )
        beta_lower, map_lower = self._node(lower, base + half, half)

        betas = []
        pmap = []
        for p in range(len(beta_lower)):
            bu = beta_upper[map_lower[p]]
            bl = beta_lower[p]
            betas.append(np.concatenate([bu ^ bl, bl]))
            pmap.append(map_upper[map_lower[p]])
        return betas, pmap

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int8)]
        self._node([llr_ch], 0, self.N)

        paths = list(zip(self.metrics, self.decisions, strict=True))
        paths.sort(key=lambda t: t[0])

        if self.crc_length > 0:
            for pm, dec in paths:
                info_bits = dec[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    return dec.astype(int), pm

        pm, dec = paths[0]
        return dec.astype(int), pm
