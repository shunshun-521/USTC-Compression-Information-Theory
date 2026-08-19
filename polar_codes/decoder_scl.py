"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
import math
from decoder_sc import (
    f_operation,
    g_operation,
    _SCNode,
    _build_polar_tree,
    _assign_frozen,
    _collect_leaves,
)


def phi_metric(pm, llr, u_bit):
    """路径度量更新（与 aff3ct phi 一致）。"""
    if u_bit == 0 and llr < 0:
        return pm - llr
    if u_bit != 0 and llr > 0:
        return pm + llr
    return pm


def crc_encode(info_bits, crc_length=8):
    """CRC 编码：CRC-8 (0x07) 或 CRC-16 (0x8005)。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & top:
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _add_parent_links(node, parent=None):
    node.parent = parent
    if node.left is not None:
        _add_parent_links(node.left, node)
        _add_parent_links(node.right, node)


def _apply_f_node(node):
    half = node.size // 2
    for i in range(half):
        node.left.lambda_arr[i] = f_operation(
            node.lambda_arr[i], node.lambda_arr[half + i]
        )


def _apply_g_node(node):
    half = node.size // 2
    for i in range(half):
        node.right.lambda_arr[i] = g_operation(
            node.lambda_arr[i],
            node.lambda_arr[half + i],
            node.left.s[i],
        )


def _compute_sums_node(node):
    half = node.size // 2
    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def _compute_depth(leaf_index, tree_depth):
    """与 aff3ct compute_depth 一致。"""
    if leaf_index == 0:
        return tree_depth - 1
    res = 0
    index = leaf_index
    while (index & 1) != 1 and res <= tree_depth - 1:
        index >>= 1
        res += 1
    return res


def _recursive_compute_llr(node, depth):
    """aff3ct recursive_compute_llr。"""
    if depth != 0:
        _recursive_compute_llr(node.parent, depth - 1)
    parent = node.parent
    if node is parent.left:
        _apply_f_node(parent)
    else:
        _apply_g_node(parent)


def _compute_llr_at_leaf(leaf, tree_depth):
    """计算叶节点 LLR。"""
    _recursive_compute_llr(leaf, _compute_depth(leaf.lane_id, tree_depth))


def _propagate_sums_from_leaf(leaf):
    """从叶节点向上传播部分和。"""
    node = leaf
    while node.parent is not None:
        parent = node.parent
        if node is parent.right:
            _compute_sums_node(parent)
        node = parent


def _duplicate_tree_llr(src_leaf, dst_leaf):
    """Lazy copy：复制 LLR 链。"""
    node_s, node_d = src_leaf, dst_leaf
    while node_s.parent is not None:
        node_s = node_s.parent
        node_d = node_d.parent
        node_d.lambda_arr[:] = node_s.lambda_arr


def _duplicate_tree_sums(src_leaf, dst_leaf):
    """复制已判决比特的部分和链。"""
    node_s, node_d = src_leaf, dst_leaf
    while node_s.parent is not None:
        node_s = node_s.parent
        node_d = node_d.parent
        if node_s.left is not None and node_d.left is not None:
            node_d.left.s[:] = node_s.left.s[:]
        node_d.s[:] = node_s.s[:]


class SCLDecoder:
    """SCL 译码器（Lazy Copy）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length

        self.trees = []
        self.leaves_list = []
        for _ in range(list_size):
            root = _build_polar_tree(self.n)
            _add_parent_links(root)
            leaves = []
            _collect_leaves(root, leaves)
            self.trees.append(root)
            self.leaves_list.append(leaves)

    def decode(self, llr_ch):
        from encoder import bit_reversal_permutation

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_work = llr_ch[br].copy()

        for tree in self.trees:
            tree.lambda_arr[:] = llr_work
            _assign_frozen(tree, self.frozen_bits)

        active_paths = {0}
        path_metrics = {0: 0.0}

        for leaf_idx in range(self.N):
            for path_id in active_paths:
                _compute_llr_at_leaf(self.leaves_list[path_id][leaf_idx], self.n)

            if self.frozen_bits[leaf_idx]:
                min_phi = float("inf")
                for path_id in active_paths:
                    lam = self.leaves_list[path_id][leaf_idx].lambda_arr[0]
                    self.leaves_list[path_id][leaf_idx].s[0] = 0
                    path_metrics[path_id] = phi_metric(path_metrics[path_id], lam, 0)
                    min_phi = min(min_phi, path_metrics[path_id])
                for path_id in active_paths:
                    path_metrics[path_id] -= min_phi
                    _propagate_sums_from_leaf(self.leaves_list[path_id][leaf_idx])
            else:
                candidates = []
                for path_id in active_paths:
                    lam = self.leaves_list[path_id][leaf_idx].lambda_arr[0]
                    pm0 = phi_metric(path_metrics[path_id], lam, 0)
                    pm1 = phi_metric(path_metrics[path_id], lam, 1)
                    candidates.append((pm0, path_id, 0))
                    candidates.append((pm1, path_id, 1))

                candidates.sort(key=lambda x: x[0])
                min_phi = candidates[0][0]

                new_active = set()
                new_metrics = {}

                if self.list_size == 1:
                    pm, path_id, bit = candidates[0]
                    self.leaves_list[path_id][leaf_idx].s[0] = bit
                    path_metrics[path_id] = pm - min_phi
                    _propagate_sums_from_leaf(self.leaves_list[path_id][leaf_idx])
                    new_active.add(path_id)
                    new_metrics[path_id] = path_metrics[path_id]
                else:
                    for i in range(len(candidates)):
                        candidates[i] = (
                            candidates[i][0] - min_phi,
                            candidates[i][1],
                            candidates[i][2],
                        )

                    used_paths = set()
                    for pm, path_id, bit in candidates:
                        if len(new_active) >= self.list_size:
                            break
                        if path_id in used_paths:
                            new_path_id = self._find_free_path(new_active)
                            self._duplicate_path(path_id, new_path_id, leaf_idx, bit, pm)
                            new_active.add(new_path_id)
                            new_metrics[new_path_id] = pm
                        else:
                            self.leaves_list[path_id][leaf_idx].s[0] = bit
                            path_metrics[path_id] = pm
                            _propagate_sums_from_leaf(
                                self.leaves_list[path_id][leaf_idx]
                            )
                            new_active.add(path_id)
                            new_metrics[path_id] = pm
                            used_paths.add(path_id)

                active_paths = new_active
                path_metrics = new_metrics

        best_path = min(active_paths, key=lambda p: path_metrics[p])
        u_hat = np.zeros(self.N, dtype=int)
        for i, leaf in enumerate(self.leaves_list[best_path]):
            u_hat[i] = leaf.s[0]

        if self.crc_length > 0:
            info_idx = np.where(self.frozen_bits == 0)[0]
            valid_paths = []
            for path_id in active_paths:
                bits = np.array(
                    [self.leaves_list[path_id][i].s[0] for i in range(self.N)]
                )
                info_bits = bits[info_idx]
                if crc_check(info_bits, self.crc_length):
                    valid_paths.append(path_id)
            if valid_paths:
                best_path = min(valid_paths, key=lambda p: path_metrics[p])

        u_hat = np.zeros(self.N, dtype=int)
        for i, leaf in enumerate(self.leaves_list[best_path]):
            u_hat[i] = leaf.s[0]

        return u_hat, path_metrics[best_path]

    def _find_free_path(self, active):
        for i in range(self.list_size):
            if i not in active:
                return i
        return self.list_size - 1

    def _duplicate_path(self, src, dst, leaf_idx, bit, pm):
        for i in range(leaf_idx):
            self.leaves_list[dst][i].s[0] = self.leaves_list[src][i].s[0]
        _duplicate_tree_sums(self.leaves_list[src][leaf_idx], self.leaves_list[dst][leaf_idx])
        if leaf_idx < self.N - 1:
            _duplicate_tree_llr(
                self.leaves_list[src][leaf_idx + 1], self.leaves_list[dst][leaf_idx + 1]
            )
        self.leaves_list[dst][leaf_idx].s[0] = bit
        _propagate_sums_from_leaf(self.leaves_list[dst][leaf_idx])
