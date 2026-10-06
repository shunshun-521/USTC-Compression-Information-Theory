"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL），基于树形因子图（Aff3ct SCL naive 流程简化版）
"""
import copy
import math

import numpy as np

from decoder_sc import (
    build_polar_tree,
    compute_depth,
    compute_llr_at_leaf,
    copy_tree_state,
    init_tree_frozen,
    propagate_sums_from_leaf,
    sc_decode,
)
from encoder import bit_reversal_permutation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """信息比特后附加 CRC 校验位"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=np.int8,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    if crc_length == 0:
        return True
    if len(bits) < crc_length:
        return False
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_penalty(llr, u):
    hard = 0 if llr >= 0 else 1
    return 0.0 if u == hard else abs(llr)


def _pm_update(pm, llr, bit):
    return pm + _pm_penalty(llr, bit)


class SCLDecoder:
    """SCL 译码器（L 棵极化树，路径度量越小越好）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._roots = []
        self._leaves_list = []
        for _ in range(list_size):
            root, leaves = build_polar_tree(N)
            init_tree_frozen(root, self.frozen_bits)
            self._roots.append(root)
            self._leaves_list.append(leaves)

    def _load_llr(self, path_id, llr):
        self._roots[path_id].lambda_[:] = llr

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        L = self.list_size
        active = {0}
        pm = {0: 0.0}
        self._load_llr(0, llr_ch)

        for leaf_idx in range(self.N):
            depth = compute_depth(leaf_idx, self.m)
            for p in list(active):
                leaf = self._leaves_list[p][leaf_idx]
                compute_llr_at_leaf(leaf, depth)

            if self.frozen_bits[leaf_idx]:
                min_pm = min(pm[p] for p in active)
                for p in active:
                    leaf = self._leaves_list[p][leaf_idx]
                    leaf.s[0] = 0
                    pm[p] = _pm_update(pm[p], leaf.lambda_[0], 0) - min_pm
                    propagate_sums_from_leaf(leaf)
            else:
                candidates = []
                for p in active:
                    leaf = self._leaves_list[p][leaf_idx]
                    llr = leaf.lambda_[0]
                    for bit in (0, 1):
                        candidates.append((pm[p] + _pm_penalty(llr, bit), p, bit))

                candidates.sort(key=lambda x: x[0])
                candidates = candidates[:L]
                new_active = set()
                new_pm = {}
                used_dst = set()

                for metric, src, bit in candidates:
                    dst = src
                    if src in new_active:
                        # 需要新路径槽
                        for cand in range(L):
                            if cand not in active and cand not in used_dst:
                                dst = cand
                                break
                        else:
                            for cand in range(L):
                                if cand not in new_active:
                                    dst = cand
                                    break
                        copy_tree_state(self._roots[src], self._roots[dst])
                        for i in range(leaf_idx):
                            self._leaves_list[dst][i].s[0] = self._leaves_list[src][i].s[0]
                    used_dst.add(dst)
                    leaf = self._leaves_list[dst][leaf_idx]
                    leaf.s[0] = bit
                    new_pm[dst] = metric
                    propagate_sums_from_leaf(leaf)
                    new_active.add(dst)

                active = new_active
                min_pm = min(new_pm.values()) if new_pm else 0.0
                pm = {k: v - min_pm for k, v in new_pm.items()}

        best_p = min(active, key=lambda p: pm.get(p, 0.0))
        u_hat = np.zeros(self.N, dtype=int)
        for i, leaf in enumerate(self._leaves_list[best_p]):
            u_hat[i] = leaf.s[0]

        if self.crc_length > 0:
            for p in sorted(active, key=lambda x: pm.get(x, 0.0)):
                u_try = np.array([lf.s[0] for lf in self._leaves_list[p]], dtype=int)
                payload = u_try[self.info_indices]
                if crc_check(payload, self.crc_length):
                    u_hat = u_try
                    best_p = p
                    break

        return u_hat, pm.get(best_p, 0.0)


def scl_decode_reordered(llr_ch, frozen_bits, list_size=4, crc_length=0):
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    dec = SCLDecoder(N, frozen_bits, list_size=list_size, crc_length=crc_length)
    return dec.decode(llr_ch[br])
