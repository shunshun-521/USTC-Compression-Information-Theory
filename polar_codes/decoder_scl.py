"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _map_channel_llr

_CRC8_POLY_BITS = [1, 0, 0, 0, 0, 0, 1, 1, 1]  # x^8 + x^2 + x + 1
_CRC16_POLY_BITS = [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1]  # CRC-16


def _gf2_crc_remainder(bits, poly_bits, crc_length):
    msg = list(map(int, bits))
    while len(msg) > crc_length:
        if msg[0]:
            for i in range(len(poly_bits)):
                msg[i] ^= poly_bits[i]
        msg = msg[1:]
    return msg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY_BITS if crc_length == 8 else _CRC16_POLY_BITS
    remainder = _gf2_crc_remainder(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]), poly, crc_length
    )
    crc_bits = np.array(remainder, dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC"""
    bits = np.asarray(bits, dtype=int).ravel()
    poly = _CRC8_POLY_BITS if crc_length == 8 else _CRC16_POLY_BITS
    remainder = _gf2_crc_remainder(bits, poly, crc_length)
    return sum(remainder) == 0


def _penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


class SCLDecoder:
    """SCL 译码器（递归极化树 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.N = N
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._metrics = [0.0]
        self._decisions = [np.zeros(N, dtype=int)]

    def decode(self, llr_ch):
        llr = _map_channel_llr(llr_ch)
        self._metrics = [0.0]
        self._decisions = [np.zeros(self.N, dtype=int)]
        self._node([llr], 0, self.N)

        paths = list(zip(self._metrics, self._decisions, strict=True))
        paths.sort(key=lambda x: x[0])

        if self.crc_length > 0:
            crc_paths = [
                (pm, dec)
                for pm, dec in paths
                if crc_check(dec[self.info_indices], self.crc_length)
            ]
            if crc_paths:
                pm, dec = crc_paths[0]
                return dec.copy(), pm

        pm, dec = paths[0]
        return dec.copy(), pm

    def _leaf(self, llrs, index):
        if self.frozen_bits[index]:
            for path, llr in enumerate(llrs):
                self._metrics[path] += _penalty(float(llr[0]), 0)
                self._decisions[path][index] = 0
            return [np.array([0], dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self._metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self._decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(path)

        self._metrics = new_metrics
        self._decisions = new_decisions
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
