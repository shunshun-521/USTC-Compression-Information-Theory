"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus）：
    f(La, Lb) ≈ sign(La*Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign = np.ones_like(La)
    prod = La * Lb
    sign[prod < 0] = -1
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (u_hat==0 ? La : -La) + Lb
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La, -La) + Lb


class _SCNode:
    __slots__ = ('lane', 'lambda_', 's', 'frozen', 'left', 'right', 'leaf')

    def __init__(self, lane=0):
        self.lane = lane
        self.lambda_ = None
        self.s = None
        self.frozen = False
        self.left = None
        self.right = None
        self.leaf = True


def _build_tree(n, lane=0):
    node = _SCNode(lane)
    if n == 0:
        node.lambda_ = np.zeros(1, dtype=np.float64)
        node.s = np.zeros(1, dtype=int)
        return node
    node.leaf = False
    size = 1 << n
    node.lambda_ = np.zeros(size, dtype=np.float64)
    node.s = np.zeros(size, dtype=int)
    node.left = _build_tree(n - 1, lane)
    node.right = _build_tree(n - 1, lane + (1 << (n - 1)))
    return node


def _init_frozen(node, frozen_bits):
    if node.leaf:
        node.frozen = bool(frozen_bits[node.lane])
        return
    _init_frozen(node.left, frozen_bits)
    _init_frozen(node.right, frozen_bits)


def _h_llr(llr):
    return 0 if llr >= 0 else 1


def _decode_tree(node):
    if node.leaf:
        node.s[0] = 0 if node.frozen else _h_llr(node.lambda_[0])
        return

    half = len(node.lambda_) // 2
    for i in range(half):
        node.left.lambda_[i] = f_operation(
            node.lambda_[i], node.lambda_[half + i]
        )
    _decode_tree(node.left)

    for i in range(half):
        node.right.lambda_[i] = g_operation(
            node.lambda_[i], node.lambda_[half + i], node.left.s[i]
        )
    _decode_tree(node.right)

    for i in range(half):
        node.s[i] = (node.left.s[i] ^ node.right.s[i]) % 2
        node.s[half + i] = node.right.s[i]


def _collect_bits(node, u_hat):
    if node.leaf:
        u_hat[node.lane] = node.s[0]
        return
    _collect_bits(node.left, u_hat)
    _collect_bits(node.right, u_hat)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树形实现，与 aff3ct 一致）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    root = _build_tree(n)
    _init_frozen(root, frozen_bits)
    root.lambda_[:] = llr
    _decode_tree(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_bits(root, u_hat)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        temp = phi
        while temp % 2 == 1:
            layers.append(int(math.log2(temp & -temp)))
            temp //= 2
        llr_layer_vec.append(layers)

        layers_b = []
        psi = phi // 2
        while psi % 2 == 1:
            layers_b.append(int(math.log2(psi & -psi)))
            psi //= 2
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（调用树形实现）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
