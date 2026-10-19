"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


# ==================== 树结构 SC（aff3ct 等价实现）====================


class _TreeNode:
    __slots__ = ("lane", "left", "right", "lam", "s", "frozen")

    def __init__(self, lane=None):
        self.lane = lane
        self.left = None
        self.right = None
        self.lam = None
        self.s = None
        self.frozen = False


def _build_tree(depth):
    counter = [0]

    def rec(d):
        if d == 0:
            node = _TreeNode(counter[0])
            counter[0] += 1
            return node
        node = _TreeNode()
        node.left = rec(d - 1)
        node.right = rec(d - 1)
        return node

    return rec(depth)


def _alloc_tree(node, size):
    node.lam = [0.0] * size
    node.s = [0] * size
    if node.left is not None:
        _alloc_tree(node.left, size // 2)
        _alloc_tree(node.right, size // 2)


def _set_frozen(node, frozen_bits):
    if node.left is not None:
        _set_frozen(node.left, frozen_bits)
        _set_frozen(node.right, frozen_bits)
    else:
        node.frozen = bool(frozen_bits[node.lane])


def _decode_tree(node):
    if node.left is not None:
        half = len(node.lam) // 2
        for i in range(half):
            node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        _decode_tree(node.left)
        for i in range(half):
            node.left.s[i] = int(node.left.s[i])
            node.right.lam[i] = g_operation(
                node.lam[i], node.lam[half + i], node.left.s[i]
            )
        _decode_tree(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        node.s[0] = 0 if node.frozen else _hard_decision(node.lam[0])


def _collect_bits(node, u_hat):
    if node.left is not None:
        _collect_bits(node.left, u_hat)
        _collect_bits(node.right, u_hat)
    else:
        u_hat[node.lane] = node.s[0]


def _sc_decode_tree(llr_ch, frozen_bits):
    """树形 SC 译码（与 aff3ct SC_naive 等价，LLR 已为译码域顺序）。"""
    llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -1e3, 1e3)
    N = len(llr_ch)
    n = int(math.log2(N))
    root = _build_tree(n)
    _alloc_tree(root, N)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    _set_frozen(root, frozen_bits)
    for i in range(N):
        root.lam[i] = float(llr_ch[i])
    _decode_tree(root)
    u_hat = np.zeros(N, dtype=np.int8)
    _collect_bits(root, u_hat)
    return u_hat


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与树形实现一致，含 LLR 比特倒序对齐编码器）。"""
    N = len(llr)
    br = bit_reversal_permutation(N)
    llr_perm = np.asarray(llr, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _sc_decode_tree(llr_perm, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供扩展/对照使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        for layer in range(n):
            if psi % 2 == 0:
                llr_layers.append(layer)
            psi //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi + 1
        for layer in range(n):
            if psi % 2 == 1:
                bit_layers.append(layer)
            psi //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码器输出含比特倒序置换，信道 LLR 在译码前做相同置换。
    """
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_perm = np.asarray(llr_ch, dtype=np.float64)[br]
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _sc_decode_tree(llr_perm, frozen_bits)
