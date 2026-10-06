"""
极化码 SC（串行抵消）译码器
树形递归实现（与 Aff3ct SC naive 一致，含部分和 s 的 XOR 回传）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 为左子树部分和向量 s）"""
    return (1 - 2 * u_hat) * La + Lb


class _SCNode:
    __slots__ = (
        "size",
        "lambda_",
        "s",
        "is_frozen",
        "left",
        "right",
        "parent",
        "is_left_child",
        "leaf_id",
    )

    def __init__(self, size):
        self.size = size
        self.lambda_ = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.is_frozen = False
        self.left = None
        self.right = None
        self.parent = None
        self.is_left_child = None
        self.leaf_id = None


def _build_tree(size, counter):
    node = _SCNode(size)
    if size == 1:
        node.leaf_id = counter[0]
        counter[0] += 1
    else:
        node.left = _build_tree(size // 2, counter)
        node.right = _build_tree(size // 2, counter)
        node.left.parent = node
        node.right.parent = node
        node.left.is_left_child = True
        node.right.is_left_child = False
    return node


def _init_frozen(node, frozen_bits):
    if node.size == 1:
        node.is_frozen = bool(frozen_bits[node.leaf_id])
    else:
        _init_frozen(node.left, frozen_bits)
        _init_frozen(node.right, frozen_bits)


def _collect_leaves(node, leaves):
    if node.size == 1:
        leaves.append(node)
    else:
        _collect_leaves(node.left, leaves)
        _collect_leaves(node.right, leaves)


def _decode_recursive(node):
    if node.size == 1:
        if node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = 1 if node.lambda_[0] < 0 else 0
        return
    half = node.size // 2
    l, r = node.left, node.right
    l.lambda_[:] = f_operation(node.lambda_[:half], node.lambda_[half:])
    _decode_recursive(l)
    r.lambda_[:] = g_operation(node.lambda_[:half], node.lambda_[half:], l.s)
    _decode_recursive(r)
    node.s[:half] = (l.s ^ r.s)[:half]
    node.s[half:] = r.s


def _sc_decode_tree(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    root = _build_tree(N, [0])
    _init_frozen(root, frozen_bits)
    root.lambda_[:] = llr
    _decode_recursive(root)
    u_hat = np.zeros(N, dtype=int)
    leaves = []
    _collect_leaves(root, leaves)
    for leaf in leaves:
        u_hat[leaf.leaf_id] = leaf.s[0]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树形参考实现）"""
    return _sc_decode_tree(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归接口：当前与树形递归实现等价（保证数值一致）"""
    return _sc_decode_tree(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """保留接口供 SCL 层向量（若使用快速 SC 时可扩展）"""
    n = int(math.log2(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        p = phi
        while p & 1:
            llr_layer_vec[phi].append(int(math.log2(p & -p)))
            p >>= 1
        p = phi
        while (p & 1) == 0 and p > 0:
            bit_layer_vec[phi].append(int(math.log2(p & -p)))
            p >>= 1
    return llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode_with_llr_reorder(llr_ch, frozen_bits):
    """信道 LLR 为码字顺序时，先做比特倒序再 SC 译码"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return sc_decode(llr_ch[br], frozen_bits)


def build_polar_tree(N):
    """构建极化码译码树（供 SCL 使用）"""
    root = _build_tree(N, [0])
    leaves = []
    _collect_leaves(root, leaves)
    return root, leaves


def apply_f_node(node):
    half = node.size // 2
    node.left.lambda_[:] = f_operation(node.lambda_[:half], node.lambda_[half:])


def apply_g_node(node):
    half = node.size // 2
    node.right.lambda_[:] = g_operation(
        node.lambda_[:half], node.lambda_[half:], node.left.s
    )


def compute_sums_node(node):
    half = node.size // 2
    node.s[:half] = (node.left.s ^ node.right.s)[:half]
    node.s[half:] = node.right.s


def compute_depth(leaf_index, tree_depth):
    if leaf_index == 0:
        return tree_depth - 1
    return int(math.log2(leaf_index & -leaf_index))


def compute_llr_at_leaf(leaf, depth):
    """沿父链计算当前叶节点的 LLR（Aff3ct recursive_compute_llr）"""
    if depth > 0 and leaf.parent is not None:
        compute_llr_at_leaf(leaf.parent, depth - 1)
    if leaf.parent is None:
        return
    if leaf.is_left_child:
        apply_f_node(leaf.parent)
    else:
        apply_g_node(leaf.parent)


def propagate_sums_from_leaf(leaf):
    """比特判决后向上回传部分和"""
    node = leaf
    while node.parent is not None and not node.is_left_child:
        compute_sums_node(node.parent)
        node = node.parent


def init_tree_frozen(root, frozen_bits):
    _init_frozen(root, frozen_bits)


def copy_tree_state(src_root, dst_root):
    """复制 lambda 与 s（用于 SCL 路径复制）"""
    if src_root.size != dst_root.size:
        raise ValueError("tree size mismatch")
    dst_root.lambda_[:] = src_root.lambda_
    dst_root.s[:] = src_root.s
    if src_root.size > 1:
        copy_tree_state(src_root.left, dst_root.left)
        copy_tree_state(src_root.right, dst_root.right)
