"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _f_exact, g_operation, _pm_penalty

CRC_POLYS = {8: 0x107, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 的 CRC 是否正确"""
    if crc_length == 0:
        return True
    poly = CRC_POLYS[crc_length]
    reg = 0
    for bit in bits:
        reg = (reg << 1) | int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 风格的路径管理）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen = frozen_bits.astype(bool)
        self.info_indices = np.where(~self.frozen)[0]
        self.list_size = list_size
        self.crc_length = crc_length
        self.metrics = [0.0]
        self.decisions = [np.zeros(N, dtype=int)]

    def _leaf(self, llrs, index):
        if self.frozen[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _pm_penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=int) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _pm_penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = self.decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=int))
            parent_map.append(path)

        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [_f_exact(llr[:half], llr[half:]) for llr in llrs]
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
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=int)]
        llr = np.asarray(llr_ch, dtype=np.float64)
        _, _ = self._node([llr], 0, self.N)

        paths = list(zip(self.metrics, self.decisions, strict=True))

        if self.crc_length > 0:
            valid = [
                (m, d)
                for m, d in paths
                if crc_check(d[self.info_indices], self.crc_length)
            ]
            if valid:
                best = min(valid, key=lambda p: p[0])
            else:
                best = min(paths, key=lambda p: p[0])
        else:
            best = min(paths, key=lambda p: p[0])

        return best[1], best[0]
