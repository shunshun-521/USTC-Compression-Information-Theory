"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import f_operation, g_operation, sc_decode_recursive


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = 0
    for b in info_bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << (crc_length) - 1))
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.concatenate([info_bits, np.array(crc_bits, dtype=int)])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(expected, bits)


def _compute_depth(index, tree_depth):
    if index == 0:
        return tree_depth - 1
    res = 0
    while (index & 1) != 1 and res <= tree_depth - 1:
        index >>= 1
        res += 1
    return res


class _SCLNode:
    __slots__ = (
        "size",
        "lambda_v",
        "s",
        "is_frozen",
        "left",
        "right",
        "father",
        "lane_id",
    )

    def __init__(self, size):
        self.size = size
        self.lambda_v = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=int)
        self.is_frozen = False
        self.left = None
        self.right = None
        self.father = None
        self.lane_id = -1

    def is_left(self):
        return self.father is not None and self.father.left is self

    def is_right(self):
        return self.father is not None and self.father.right is self


def _build_scl_tree(size, lane_counter):
    node = _SCLNode(size)
    if size > 1:
        half = size // 2
        node.left = _build_scl_tree(half, lane_counter)
        node.right = _build_scl_tree(half, lane_counter)
        node.left.father = node
        node.right.father = node
    else:
        node.lane_id = lane_counter[0]
        lane_counter[0] += 1
    return node


def _collect_leaves(node, leaves):
    if node.size == 1:
        leaves.append(node)
    else:
        _collect_leaves(node.left, leaves)
        _collect_leaves(node.right, leaves)


def _init_frozen_scl(node, frozen_bits):
    if node.size == 1:
        node.is_frozen = bool(frozen_bits[node.lane_id])
    else:
        _init_frozen_scl(node.left, frozen_bits)
        _init_frozen_scl(node.right, frozen_bits)


def _apply_f(node):
    half = node.size // 2
    for i in range(half):
        node.left.lambda_v[i] = f_operation(
            node.lambda_v[i], node.lambda_v[half + i]
        )


def _apply_g(node):
    half = node.size // 2
    for i in range(half):
        node.right.lambda_v[i] = g_operation(
            node.lambda_v[i], node.lambda_v[half + i], node.left.s[i]
        )


def _compute_sums(node):
    half = node.size // 2
    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def _recursive_compute_llr(leaf, depth):
    if depth != 0:
        _recursive_compute_llr(leaf.father, depth - 1)
    father = leaf.father
    if father is None:
        return
    if leaf.is_left():
        _apply_f(father)
    if leaf.is_right():
        _apply_g(father)


def _propagate_sums(leaf):
    node = leaf
    if not node.father:
        return
    _compute_sums(node.father)
    if node.is_right():
        _propagate_sums(node.father)


def _path_metric_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    return pm + (0.0 if u == u_hard else abs(llr))


def _duplicate_tree_llr(node_a, node_b):
    node_b.lambda_v[:] = node_a.lambda_v[:]
    if node_a.father is not None:
        _duplicate_tree_llr(node_a.father, node_b.father)


def _duplicate_tree_sums(node_a, node_b, stop_lane):
    node_b.s[:] = node_a.s[:]
    if node_a.lane_id != stop_lane and node_a.father is not None:
        _duplicate_tree_sums(node_a.father, node_b.father, stop_lane)


class _PolarTreePath:
    def __init__(self, N, frozen_bits):
        self.N = N
        self.m = int(np.log2(N))
        lane = [0]
        self.root = _build_scl_tree(N, lane)
        self.leaves = []
        _collect_leaves(self.root, self.leaves)
        _init_frozen_scl(self.root, frozen_bits)
        self.pm = 0.0


class SCLDecoder:
    """SCL 译码器（aff3ct naive SCL 结构，L 棵预分配树）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.m = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.paths = [_PolarTreePath(N, self.frozen_bits) for _ in range(list_size)]

    def _load(self, llr_ch):
        for p in self.paths:
            p.root.lambda_v[:] = llr_ch
            p.pm = 0.0
        self.active_paths = {0}

    def _duplicate_path(self, src_idx, leaf_index):
        free_idx = None
        for i in range(self.list_size):
            if i not in self.active_paths:
                free_idx = i
                break
        if free_idx is None:
            return None
        src_leaf = self.paths[src_idx].leaves[leaf_index]
        dst_leaf = self.paths[free_idx].leaves[leaf_index]
        _duplicate_tree_llr(src_leaf, dst_leaf)
        _duplicate_tree_sums(src_leaf, dst_leaf, src_leaf.lane_id)
        self.paths[free_idx].pm = self.paths[src_idx].pm
        self.active_paths.add(free_idx)
        return free_idx

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1 and self.crc_length == 0:
            u_hat = sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        self._load(llr_ch)

        for leaf_index in range(self.N):
            for path in list(self.active_paths):
                leaf = self.paths[path].leaves[leaf_index]
                depth = _compute_depth(leaf_index, self.m)
                _recursive_compute_llr(leaf, depth)

            if self.frozen_bits[leaf_index]:
                min_phi = float("inf")
                for path in self.active_paths:
                    leaf = self.paths[path].leaves[leaf_index]
                    leaf.s[0] = 0
                    self.paths[path].pm = _path_metric_update(
                        self.paths[path].pm, leaf.lambda_v[0], 0
                    )
                    min_phi = min(min_phi, self.paths[path].pm)
                for path in self.active_paths:
                    self.paths[path].pm -= min_phi
            else:
                metrics = []
                for path in self.active_paths:
                    leaf = self.paths[path].leaves[leaf_index]
                    llr = leaf.lambda_v[0]
                    pm = self.paths[path].pm
                    metrics.append((path, 0, _path_metric_update(pm, llr, 0)))
                    metrics.append((path, 1, _path_metric_update(pm, llr, 1)))

                min_phi = min(m[2] for m in metrics)
                metrics = [(p, u, pm - min_phi) for p, u, pm in metrics]
                metrics.sort(key=lambda x: x[2])

                if len(self.active_paths) <= self.list_size // 2:
                    last = list(self.active_paths)
                    for path in last:
                        self._duplicate_path(path, leaf_index)
                else:
                    keep = metrics[: self.list_size]
                    new_active = set()
                    used_dup = set()
                    for path, u, pm in keep:
                        if path in new_active:
                            dup = self._duplicate_path(path, leaf_index)
                            if dup is not None:
                                path = dup
                        elif path in used_dup:
                            dup = self._duplicate_path(path, leaf_index)
                            if dup is not None:
                                path = dup
                        else:
                            used_dup.add(path)
                        leaf = self.paths[path].leaves[leaf_index]
                        leaf.s[0] = u
                        self.paths[path].pm = pm
                        new_active.add(path)
                    self.active_paths = new_active

            for path in self.active_paths:
                _propagate_sums(self.paths[path].leaves[leaf_index])

        best_crc = None
        best_pm = None
        for path in self.active_paths:
            u_hat = np.array([leaf.s[0] for leaf in self.paths[path].leaves], dtype=int)
            pm = self.paths[path].pm
            if self.crc_length > 0:
                payload = u_hat[self.info_indices]
                if crc_check(payload, self.crc_length):
                    if best_crc is None or pm < best_crc[1]:
                        best_crc = (u_hat, pm)
            if best_pm is None or pm < best_pm[1]:
                best_pm = (u_hat, pm)

        chosen = best_crc if best_crc is not None else best_pm
        return chosen[0].copy(), chosen[1]
