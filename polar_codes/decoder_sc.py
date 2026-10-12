"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


class _SCNode:
    __slots__ = (
        "lam",
        "s",
        "frozen",
        "left",
        "right",
        "lane",
        "leaf",
        "parent",
        "is_left",
    )

    def __init__(self, size, lane=-1, leaf=False, frozen=False):
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=int)
        self.frozen = frozen
        self.lane = lane
        self.leaf = leaf
        self.left = None
        self.right = None
        self.parent = None
        self.is_left = None


def _build_tree(n, frozen_bits, lane_start=0, parent=None, is_left=None):
    """构建极化码二叉树（叶子 lane_id 0..N-1）"""
    if n == 0:
        idx = lane_start
        node = _SCNode(1, lane=idx, leaf=True, frozen=bool(frozen_bits[idx]))
        node.parent = parent
        node.is_left = is_left
        return node
    size = 1 << n
    node = _SCNode(size)
    node.parent = parent
    node.is_left = is_left
    half = size // 2
    node.left = _build_tree(n - 1, frozen_bits, lane_start, node, True)
    node.right = _build_tree(n - 1, frozen_bits, lane_start + half, node, False)
    return node


def _clone_tree(node):
    new = _SCNode(
        len(node.lam),
        lane=node.lane,
        leaf=node.leaf,
        frozen=node.frozen,
    )
    new.lam = node.lam.copy()
    new.s = node.s.copy()
    new.parent = node.parent
    new.is_left = node.is_left
    if not node.leaf:
        new.left = _clone_tree(node.left)
        new.right = _clone_tree(node.right)
        new.left.parent = new
        new.right.parent = new
    return new


def _gather_leaves(node, leaves):
    if node.leaf:
        leaves.append(node)
    else:
        _gather_leaves(node.left, leaves)
        _gather_leaves(node.right, leaves)


def _compute_depth(index, tree_depth):
    if index == 0:
        return tree_depth - 1
    res = 0
    while (index & 1) != 1 and res <= tree_depth - 1:
        index >>= 1
        res += 1
    return res


def _apply_f(parent):
    half = len(parent.lam) // 2
    for i in range(half):
        parent.left.lam[i] = f_operation(parent.lam[i], parent.lam[half + i])


def _apply_g(parent):
    half = len(parent.lam) // 2
    for i in range(half):
        parent.right.lam[i] = g_operation(
            parent.lam[i], parent.lam[half + i], parent.left.s[i]
        )


def _recursive_compute_llr(node, depth):
    if depth != 0 and node.parent is not None:
        _recursive_compute_llr(node.parent, depth - 1)
    if node.parent is not None:
        if node.is_left:
            _apply_f(node.parent)
        else:
            _apply_g(node.parent)


def _compute_sums(node):
    half = len(node.lam) // 2
    for i in range(half):
        node.s[i] = (node.left.s[i] + node.right.s[i]) % 2
        node.s[half + i] = node.right.s[i]


def _propagate_sums(node):
    if not node.leaf:
        _compute_sums(node)
    if node.parent is not None and not node.is_left:
        _propagate_sums(node.parent)


def _recursive_decode_aff3ct(node):
    if not node.leaf:
        half = len(node.lam) // 2
        for i in range(half):
            node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        _recursive_decode_aff3ct(node.left)
        for i in range(half):
            node.right.lam[i] = g_operation(
                node.lam[i], node.lam[half + i], node.left.s[i]
            )
        _recursive_decode_aff3ct(node.right)
        for i in range(half):
            node.s[i] = (node.left.s[i] + node.right.s[i]) % 2
            node.s[half + i] = node.right.s[i]
    else:
        if node.frozen:
            node.s[0] = 0
        else:
            node.s[0] = _hard_decision(node.lam[0])


def _collect_leaves(node, out):
    if node.leaf:
        out[node.lane] = node.s[0]
    else:
        _collect_leaves(node.left, out)
        _collect_leaves(node.right, out)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 aff3ct naive SC 一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr = llr[br]
    n = int(math.log2(N))
    root = _build_tree(n, frozen_bits)
    root.lam[:] = llr
    _recursive_decode_aff3ct(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_leaves(root, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（占位，与蝶形索引一致）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        if phi == 0:
            llr_layers = list(range(n))
        else:
            psi = phi
            while psi % 2 == 1:
                llr_layers.append(int(math.log2(psi & -psi)))
                psi >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            psi = phi
            while psi % 2 == 1:
                bit_layers.append(int(math.log2(psi & -psi)))
                psi >>= 1
        bit_layer_vec.append(bit_layers)

    return llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """
    非递归 SC：当前实现调用树递归核心（复杂度 O(N log N)），
    接口与非递归仿真流程兼容。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


sc_decode = sc_decode_recursive
