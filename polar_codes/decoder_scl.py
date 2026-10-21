"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation

# CRC-8: 0x07, CRC-16: 0x8005
_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8 if crc_length == 8 else 1):
            if crc_length == 8:
                if reg & 0x80:
                    reg = ((reg << 1) ^ (poly << (crc_length - 8))) & 0xFF
                else:
                    reg = (reg << 1) & 0xFF
            else:
                break
    if crc_length == 16:
        for b in bits:
            reg ^= int(b) << 15
            if reg & 0x8000:
                reg = ((reg << 1) ^ poly) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = _CRC8_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 7
            for _ in range(8):
                if rem & 0x80:
                    rem = ((rem << 1) ^ poly) & 0xFF
                else:
                    rem = (rem << 1) & 0xFF
        crc_bits = np.array([(rem >> (7 - i)) & 1 for i in range(8)], dtype=np.int8)
    elif crc_length == 16:
        poly = _CRC16_POLY
        rem = 0
        for b in info_bits:
            rem ^= int(b) << 15
            for _ in range(16):
                if rem & 0x8000:
                    rem = ((rem << 1) ^ poly) & 0xFFFF
                else:
                    rem = (rem << 1) & 0xFFFF
        crc_bits = np.array([(rem >> (15 - i)) & 1 for i in range(16)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC"""
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class SCLDecoder:
    """SCL 译码器（Lazy Copy 通过路径索引映射实现）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = int(N)
        self.n = int(np.log2(N))
        fb = np.asarray(frozen_bits, dtype=np.int8)
        self.frozen = fb.astype(bool)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.info_indices = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.int8)]
        paths_llr = [llr_ch.copy()]

        def leaf(llrs, index):
            nonlocal metrics, decisions
            if self.frozen[index]:
                for p, llr in enumerate(llrs):
                    metrics[p] += _penalty(float(llr[0]), 0)
                    decisions[p][index] = 0
                return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

            candidates = []
            for p, llr in enumerate(llrs):
                for bit in (0, 1):
                    candidates.append((metrics[p] + _penalty(float(llr[0]), bit), p, bit))
            candidates.sort(key=lambda x: x[0])
            kept = candidates[: self.list_size]

            new_metrics, new_decisions, betas, parent_map = [], [], [], []
            for metric, p, bit in kept:
                new_metrics.append(metric)
                d = decisions[p].copy()
                d[index] = bit
                new_decisions.append(d)
                betas.append(np.array([bit], dtype=np.int8))
                parent_map.append(p)
            metrics = new_metrics
            decisions = new_decisions
            return betas, parent_map

        def node(llrs, base, length):
            if length == 1:
                return leaf(llrs, base)

            half = length // 2
            upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
            beta_u, map_u = node(upper, base, half)

            lower = [
                g_operation(llrs[map_u[p]][:half], llrs[map_u[p]][half:], beta_u[p])
                for p in range(len(map_u))
            ]
            beta_l, map_l = node(lower, base + half, half)

            beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
            betas = [
                np.concatenate([(beta_u[p] ^ beta_l[p]) % 2, beta_l[p]])
                for p in range(len(beta_l))
            ]
            parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
            return betas, parent_map

        _, _ = node([llr_ch], 0, self.N)

        best_idx = 0
        if self.crc_length > 0:
            valid = []
            for p, d in enumerate(decisions):
                info_bits = d[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append((metrics[p], p))
            if valid:
                valid.sort(key=lambda x: x[0])
                best_idx = valid[0][1]

        u_hat = decisions[best_idx].astype(int)
        return u_hat, metrics[best_idx]
