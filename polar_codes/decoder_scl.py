"""
极化码 SCL（串行抵消列表）译码器，支持 CRC 辅助 CA-SCL
"""
import numpy as np


def f_operation(La, Lb):
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


# CRC-8 (0x07), CRC-16 (0x8005)
_CRC_POLY = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    poly = _CRC_POLY[crc_length]
    reg = 0
    for b in np.asarray(info_bits, dtype=int):
        reg ^= (b << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return np.concatenate([info_bits, np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)])


def crc_check(bits, crc_length=8):
    poly = _CRC_POLY[crc_length]
    data = np.asarray(bits, dtype=int)
    reg = 0
    for b in data:
        reg ^= (b << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


class SCLDecoder:
    """SCL 译码器（单路径时等价 SC）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        if self.frozen.dtype != bool:
            self.frozen = self.frozen.astype(bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = self._scl_decode(llr_ch)
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p[1][self.info_idx], self.crc_length)]
            if valid:
                paths = valid
        u_hat = paths[0][1]
        pm = paths[0][0]
        return u_hat.astype(int), pm

    def _scl_decode(self, channel_llr):
        metrics = [0.0]
        decisions = [np.zeros(self.N, dtype=np.uint8)]
        codewords, _ = self._node([channel_llr], 0, self.N, metrics, decisions)
        paths = list(zip(metrics, decisions, codewords))
        return sorted(paths, key=lambda x: x[0])

    def _leaf(self, llrs, index, metrics, decisions):
        if self.frozen[index]:
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
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.uint8))
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
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = self._node(lower, base + half, half, metrics, decisions)

        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map
