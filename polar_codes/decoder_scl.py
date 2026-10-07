"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation_exact, g_operation, _to_tree_frozen


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if hard == bit else abs(float(llr))


class SCLDecoder:
    """SCL 译码器（list_size=1 时等价于 SC）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_tree = _to_tree_frozen(frozen_bits, N)
        self.list_size = int(list_size)
        self.crc_length = int(crc_length)
        self.info_idx = np.flatnonzero(np.asarray(frozen_bits, dtype=int) == 0)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        self.metrics = [0.0]
        self.decisions = [np.zeros(self.N, dtype=np.uint8)]
        self._node([llr], 0, self.N)
        paths = sorted(zip(self.metrics, self.decisions), key=lambda x: x[0])

        if self.crc_length > 0:
            good = []
            for pm, u_tree in paths:
                if crc_check(u_tree[self.info_idx], self.crc_length):
                    good.append((pm, u_tree))
            if good:
                paths = good

        pm, u_tree = paths[0]
        return u_tree.astype(int), pm

    def _leaf(self, llrs, index):
        if self.frozen_tree[index]:
            for path, llr in enumerate(llrs):
                self.metrics[path] += _penalty(float(llr[0]), 0)
                self.decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (self.metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[: self.list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = self.decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)

        self.metrics = new_metrics
        self.decisions = new_decisions
        return betas, parent_map

    def _node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation_exact(l[:half], l[half:]) for l in llrs]
        beta_u, map_u = self._node(upper, base, half)

        a = [llrs[map_u[p]][:half] for p in range(len(map_u))]
        b = [llrs[map_u[p]][half:] for p in range(len(map_u))]
        lower = [
            g_operation(a[p], b[p], beta_u[p].astype(np.float64)) for p in range(len(map_u))
        ]
        beta_l, map_l = self._node(lower, base + half, half)

        beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
        betas = [
            np.concatenate([beta_u[p] ^ beta_l[p], beta_l[p]]) for p in range(len(map_l))
        ]
        parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
        return betas, parent_map
