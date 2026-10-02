"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, _frozen_mask


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_step(reg, bit, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg ^= int(bit) << (crc_length - 1)
    if reg & top:
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg = _crc_step(reg, b, poly, crc_length)
    return reg


def _crc_encode_bits(info_bits, poly, crc_length):
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    reg = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = []
    for _ in range(crc_length):
        crc_bits.append(1 if reg & top else 0)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    return crc_bits


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    crc_bits = np.array(_crc_encode_bits(info_bits, poly, crc_length), dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器（树递归 + 路径度量）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = _frozen_mask(frozen_bits)
        self.L = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int8)]

        def leaf(llrs, index):
            nonlocal metrics, decisions
            if self.frozen[index]:
                for p, llr in enumerate(llrs):
                    metrics[p] += _penalty(float(llr[0]), 0)
                    decisions[p][index] = 0
                return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

            cands = [
                (metrics[p] + _penalty(float(llr[0]), bit), p, bit)
                for p, llr in enumerate(llrs)
                for bit in (0, 1)
            ]
            cands.sort(key=lambda x: x[0])
            kept = cands[: self.L]
            new_metrics, new_decisions, betas, parent_map = [], [], [], []
            for metric, path, bit in kept:
                new_metrics.append(metric)
                dec = decisions[path].copy()
                dec[index] = bit
                new_decisions.append(dec)
                betas.append(np.array([bit], dtype=np.int8))
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

        _, _ = node([llr_ch], 0, self.N)
        paths = list(zip(metrics, decisions, strict=True))
        paths.sort(key=lambda x: x[0])

        for pm, u_hat in paths:
            if self.crc_length > 0:
                payload = u_hat[self.info_positions]
                if not crc_check(payload, self.crc_length):
                    continue
            return u_hat.astype(int), pm

        best_pm, best_u = paths[0]
        return best_u.astype(int), best_pm
