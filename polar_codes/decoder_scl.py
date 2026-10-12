"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import (
    sc_decode_recursive,
    _build_tree,
    _clone_tree,
    _gather_leaves,
    _compute_depth,
    _recursive_compute_llr,
    _propagate_sums,
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
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int).ravel()
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


def _pm_update(pm, llr, u):
    u_hat = 0 if llr >= 0 else 1
    return pm + (0.0 if u == u_hat else abs(llr))


class SCLDecoder:
    """SCL 译码器（树结构 + 路径复制）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._leaves_template = []
        root = _build_tree(self.n, self.frozen_bits)
        _gather_leaves(root, self._leaves_template)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u, 0.0

        br = bit_reversal_permutation(self.N)
        llr = llr_ch[br]

        roots = [_build_tree(self.n, self.frozen_bits)]
        roots[0].lam[:] = llr
        paths = [(0.0, roots[0])]

        for leaf_idx in range(self.N):
            leaf_ref = self._leaves_template[leaf_idx]
            depth = _compute_depth(leaf_idx, self.n)

            for pm, root in paths:
                leaf = self._leaves_template[leaf_idx]
                # 定位当前路径树中的对应叶子（按 lane 匹配）
                cur_leaf = self._find_leaf(root, leaf_ref.lane)
                _recursive_compute_llr(cur_leaf, depth)

            if leaf_ref.frozen:
                new_paths = []
                for pm, root in paths:
                    cur_leaf = self._find_leaf(root, leaf_ref.lane)
                    pm2 = _pm_update(pm, cur_leaf.lam[0], 0)
                    cur_leaf.s[0] = 0
                    _propagate_sums(cur_leaf)
                    new_paths.append((pm2, root))
                paths = new_paths
            else:
                candidates = []
                for pm, root in paths:
                    cur_leaf = self._find_leaf(root, leaf_ref.lane)
                    llr_bit = cur_leaf.lam[0]
                    for u in (0, 1):
                        root_copy = _clone_tree(root)
                        leaf_copy = self._find_leaf(root_copy, leaf_ref.lane)
                        pm2 = _pm_update(pm, llr_bit, u)
                        leaf_copy.s[0] = u
                        _propagate_sums(leaf_copy)
                        candidates.append((pm2, root_copy))
                candidates.sort(key=lambda x: x[0])
                paths = candidates[: self.list_size]

        paths.sort(key=lambda x: x[0])
        best_pm, best_root = paths[0]
        u_hat = np.zeros(self.N, dtype=int)
        for leaf in self._leaves_template:
            lf = self._find_leaf(best_root, leaf.lane)
            u_hat[leaf.lane] = lf.s[0]

        if self.crc_length > 0:
            valid = []
            for pm, root in paths:
                u_try = np.zeros(self.N, dtype=int)
                for leaf in self._leaves_template:
                    lf = self._find_leaf(root, leaf.lane)
                    u_try[leaf.lane] = lf.s[0]
                if crc_check(u_try[self.info_indices], self.crc_length):
                    valid.append((pm, u_try))
            if valid:
                best_pm, u_hat = min(valid, key=lambda x: x[0])

        return u_hat, best_pm

    def _find_leaf(self, root, lane):
        stack = [root]
        while stack:
            node = stack.pop()
            if node.leaf:
                if node.lane == lane:
                    return node
            else:
                stack.append(node.right)
                stack.append(node.left)
        raise ValueError(f"lane {lane} not found")
