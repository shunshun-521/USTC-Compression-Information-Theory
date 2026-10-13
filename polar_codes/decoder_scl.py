"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from polar_tree import (
    PolarTree,
    compute_depth,
    metric_update,
    propagate_sums,
    recursive_compute_llr,
    tree_sc_decode,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器（多路径极化树）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.trees = [PolarTree(N, frozen_bits) for _ in range(list_size)]

    def decode(self, llr_ch):
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = tree_sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        L = self.list_size
        trees = self.trees
        trees[0].load(llr_ch)
        active = {0}
        free_slots = set(range(1, L))

        for leaf_index in range(self.N):
            for p in active:
                recursive_compute_llr(
                    trees[p].leaves[leaf_index], compute_depth(leaf_index, self.m)
                )

            if self.frozen_bits[leaf_index]:
                min_pm = float("inf")
                for p in active:
                    leaf = trees[p].leaves[leaf_index]
                    leaf.s[0] = 0
                    trees[p].pm = metric_update(trees[p].pm, leaf.lam[0], 0)
                    min_pm = min(min_pm, trees[p].pm)
                for p in active:
                    trees[p].pm -= min_pm
                for p in active:
                    propagate_sums(trees[p].leaves[leaf_index])
            else:
                metrics = []
                for p in active:
                    leaf = trees[p].leaves[leaf_index]
                    llr0 = leaf.lam[0]
                    metrics.append((p, 0, metric_update(trees[p].pm, llr0, 0)))
                    metrics.append((p, 1, metric_update(trees[p].pm, llr0, 1)))
                metrics.sort(key=lambda x: x[2])
                metrics = metrics[:L]

                next_active = set()
                for src, bit, pm in metrics:
                    if src in next_active:
                        if free_slots:
                            slot = free_slots.pop()
                        else:
                            slot = min(
                                set(range(L)) - next_active,
                                key=lambda s: trees[s].pm,
                            )
                        trees[slot] = trees[src].clone()
                        dst = slot
                    else:
                        dst = src
                    leaf = trees[dst].leaves[leaf_index]
                    leaf.s[0] = bit
                    trees[dst].pm = pm
                    propagate_sums(leaf)
                    next_active.add(dst)
                active = next_active

        info_positions = np.where(~self.frozen_bits)[0]
        if self.crc_length > 0:
            valid = [
                p
                for p in active
                if crc_check(trees[p].u_hat()[info_positions], self.crc_length)
            ]
            best = min(valid if valid else active, key=lambda p: trees[p].pm)
        else:
            best = min(active, key=lambda p: trees[p].pm)

        return trees[best].u_hat(), trees[best].pm
