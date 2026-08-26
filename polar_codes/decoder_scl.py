"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from decoder_sc import f_operation, g_operation, _penalty

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_update(crc, bit, crc_length, poly):
    fb = (int(bit) ^ ((crc >> (crc_length - 1)) & 1)) & 1
    mask = (1 << crc_length) - 1
    crc = (crc << 1) & mask
    if fb:
        crc ^= poly
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc = 0
    for bit in info_bits:
        crc = _crc_update(crc, bit, crc_length, poly)
    crc_bits = np.array(
        [(crc >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    crc = 0
    for bit in bits:
        crc = _crc_update(crc, bit, crc_length, poly)
    return crc == 0


class SCLDecoder:
    """SCL 译码器（极化树递归 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int8)]
        codewords, _ = self._node([llr], 0, self.N, metrics, decisions)
        paths = sorted(
            zip(metrics, decisions, codewords, strict=True), key=lambda p: p[0]
        )

        crc_valid = []
        for metric, u_hat, _ in paths:
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    crc_valid.append((metric, u_hat))
            else:
                crc_valid.append((metric, u_hat))

        pool = crc_valid if crc_valid else [(p[0], p[1]) for p in paths]
        best_metric, best_u = min(pool, key=lambda x: x[0])
        return best_u.copy(), best_metric

    def _leaf(self, llrs, index, metrics, decisions):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                metrics[path] += _penalty(float(llr[0]), 0)
                decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[:self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(path)

        metrics[:] = new_metrics
        decisions[:] = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length, metrics, decisions):
        if length == 1:
            return self._leaf(llrs, base, metrics, decisions)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper, map_upper = self._node(upper, base, half, metrics, decisions)

        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [
            g_operation(a[p], b[p], beta_upper[p].astype(np.float64))
            for p in range(len(beta_upper))
        ]
        beta_lower, map_lower = self._node(lower, base + half, half, metrics, decisions)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
