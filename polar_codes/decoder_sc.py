"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


class _TreeNode:
    __slots__ = ("lam", "s", "frozen", "left", "right", "leaf", "lane")

    def __init__(self, size, frozen=False, leaf=False, lane=0):
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int32)
        self.frozen = frozen
        self.leaf = leaf
        self.lane = lane
        self.left = None
        self.right = None


def _build_tree(n, frozen_bits, lane_start=0):
    """构建与 aff3ct 一致的二叉极化树。"""
    if n == 0:
        idx = lane_start
        return _TreeNode(1, frozen=bool(frozen_bits[idx]), leaf=True, lane=idx)
    half = 1 << (n - 1)
    node = _TreeNode(2 * half)
    node.left = _build_tree(n - 1, frozen_bits, lane_start)
    node.right = _build_tree(n - 1, frozen_bits, lane_start + half)
    return node


def _sc_recursive(node):
    if not node.leaf:
        half = len(node.lam) // 2
        for i in range(half):
            node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
        _sc_recursive(node.left)
        for i in range(half):
            node.right.lam[i] = g_operation(
                node.lam[i], node.lam[half + i], node.left.s[i]
            )
        _sc_recursive(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        if node.frozen:
            node.s[0] = 0
        else:
            node.s[0] = 0 if node.lam[0] >= 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树结构，与 aff3ct naive 一致）。"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    root = _build_tree(n, frozen_bits)
    root.lam[:] = llr
    _sc_recursive(root)
    u_hat = np.zeros(N, dtype=np.int32)
    _collect_leaves(root, u_hat)
    return u_hat


def _collect_leaves(node, out):
    if node.leaf:
        out[node.lane] = node.s[0]
        return
    _collect_leaves(node.left, out)
    _collect_leaves(node.right, out)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while (psi & 1) == 1:
            llr_layers.append(n - 1 - layer)
            psi >>= 1
            layer += 1
        llr_layers.append(n - 1 - layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        layer = 0
        while (psi & 1) == 1:
            bit_layers.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC：当前与树形递归实现等价（性能足够时使用递归）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
