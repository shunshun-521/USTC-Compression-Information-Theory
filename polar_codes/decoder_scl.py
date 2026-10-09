"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import f_operation, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        rem = _crc8_encode(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(7, -1, -1)], dtype=np.int8)
    elif crc_length == 16:
        rem = _crc16_encode(info_bits)
        crc_bits = np.array([(rem >> i) & 1 for i in range(15, -1, -1)], dtype=np.int8)
    else:
        raise ValueError("crc_length must be 8 or 16")
    return np.concatenate([info_bits, crc_bits])


def _crc8_encode(info_bits):
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 7
        for _ in range(8):
            if reg & 0x80:
                reg = ((reg << 1) ^ CRC8_POLY) & 0xFF
            else:
                reg = (reg << 1) & 0xFF
    return reg


def _crc16_encode(info_bits):
    reg = 0
    for b in info_bits:
        reg ^= int(b) << 15
        for _ in range(16):
            if reg & 0x8000:
                reg = ((reg << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                reg = (reg << 1) & 0xFFFF
    return reg


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    payload = bits[:-crc_length]
    return np.array_equal(bits[-crc_length:], crc_encode(payload, crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径在叶节点分裂）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int8)]

    def _leaf(self, llrs, index):
        if self.frozen_bits[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, llr in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((self.metrics[path] + _penalty(float(llr[0]), bit), path, bit))
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.L_size]

        new_metrics = []
        new_decisions = []
        betas = []
        parent_map = []
        for pm, path, bit in kept:
            new_metrics.append(pm)
            d = self.decisions[path].copy()
            d[index] = bit
            new_decisions.append(d)
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
        beta_u, map_u = self._node(upper, base, half)
        a = [llrs[map_u[p]][:half] for p in range(len(map_u))]
        b = [llrs[map_u[p]][half:] for p in range(len(map_u))]
        lower = [g_operation(a[p], b[p], beta_u[p]) for p in range(len(beta_u))]
        beta_l, map_l = self._node(lower, base + half, half)
        beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
        betas = [
            np.concatenate([(beta_u[p] ^ beta_l[p]) % 2, beta_l[p]]) for p in range(len(map_l))
        ]
        parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
        return betas, parent_map

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.int8)]
        self._node([llr_ch], 0, self.N)

        best_i = int(np.argmin(self.metrics))
        u_hat = self.decisions[best_i].copy()

        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [
                (self.metrics[i], self.decisions[i])
                for i in range(len(self.decisions))
                if crc_check(self.decisions[i][info_positions], self.crc_length)
            ]
            if valid:
                valid.sort(key=lambda x: x[0])
                u_hat = valid[0][1].copy()
                return u_hat, valid[0][0]

        return u_hat, self.metrics[best_i]
