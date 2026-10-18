"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_val = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC（bits 含信息位+CRC）"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器（树形递归 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits).astype(bool)
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.flatnonzero(~self.frozen_bits)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl_decode(llr_ch)
        paths.sort(key=lambda p: p[0])

        if self.crc_length > 0:
            for pm, u_hat in paths:
                payload = u_hat[self.info_positions]
                if crc_check(payload, self.crc_length):
                    return u_hat.astype(int), pm
        pm, u_hat = paths[0]
        return u_hat.astype(int), pm

    def _scl_decode(self, channel_llr):
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.uint8)]

        def leaf(llrs, index):
            nonlocal metrics, decisions
            if self.frozen_bits[index]:
                for path, llr in enumerate(llrs):
                    metrics[path] += _penalty(float(llr[0]), 0)
                    decisions[path][index] = 0
                return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

            candidates = [
                (metrics[path] + _penalty(float(llr[0]), bit), path, bit)
                for path, llr in enumerate(llrs)
                for bit in (0, 1)
            ]
            candidates.sort(key=lambda c: c[0])
            kept = candidates[: self.L]

            new_metrics, new_decisions, betas, parent_map = [], [], [], []
            for metric, path, bit in kept:
                new_metrics.append(metric)
                dec = decisions[path].copy()
                dec[index] = bit
                new_decisions.append(dec)
                betas.append(np.array([bit], dtype=np.uint8))
                parent_map.append(path)
            metrics = new_metrics
            decisions = new_decisions
            return betas, parent_map

        def node(llrs, base, length):
            if length == 1:
                return leaf(llrs, base)

            half = length // 2
            upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
            beta_u, map_u = node(upper, base, half)

            a = [llrs[map_u[p]][:half] for p in range(len(map_u))]
            b = [llrs[map_u[p]][half:] for p in range(len(map_u))]
            lower = [g_operation(a[p], b[p], beta_u[p]) for p in range(len(beta_u))]
            beta_l, map_l = node(lower, base + half, half)

            beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
            betas = [
                np.concatenate([beta_u[p] ^ beta_l[p], beta_l[p]])
                for p in range(len(beta_l))
            ]
            parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
            return betas, parent_map

        codewords, _ = node([channel_llr], 0, self.N)
        return list(zip(metrics, decisions, strict=True))
