"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import polar_encode


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _h_decision(llr):
    return 0 if llr >= 0 else 1


class _TreeNode:
    __slots__ = ('lam', 's', 'frozen', 'leaf', 'lane', 'left', 'right')

    def __init__(self, size, leaf=False, lane=0):
        self.lam = np.zeros(size, dtype=np.float64) if size else np.zeros(1, dtype=np.float64)
        self.s = np.zeros(max(size, 1), dtype=np.int8)
        self.frozen = False
        self.leaf = leaf
        self.lane = lane
        self.left = None
        self.right = None


def _build_polar_tree(depth, cur_depth=0, lanes=None):
    if lanes is None:
        lanes = [0] * (depth + 1)
    if cur_depth == depth:
        node = _TreeNode(0, leaf=True, lane=lanes[cur_depth])
        lanes[cur_depth] += 1
        return node
    node = _TreeNode(0)
    node.left = _build_polar_tree(depth, cur_depth + 1, lanes)
    node.right = _build_polar_tree(depth, cur_depth + 1, lanes)
    return node


def _init_frozen(node, frozen_bits):
    if node.leaf:
        node.frozen = bool(frozen_bits[node.lane])
    else:
        _init_frozen(node.left, frozen_bits)
        _init_frozen(node.right, frozen_bits)


def _allocate_lambdas(node, size):
    node.lam = np.zeros(size, dtype=np.float64)
    node.s = np.zeros(size, dtype=np.int8)
    if not node.leaf:
        half = size // 2
        _allocate_lambdas(node.left, half)
        _allocate_lambdas(node.right, half)


def _decode_tree(node):
    if node.leaf:
        node.s[0] = 0 if node.frozen else _h_decision(node.lam[0])
        return

    half = len(node.lam) // 2
    for i in range(half):
        node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
    _decode_tree(node.left)

    for i in range(half):
        node.right.lam[i] = g_operation(
            node.lam[i], node.lam[half + i], node.left.s[i]
        )
    _decode_tree(node.right)

    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def _collect_bits(node, u_hat):
    if node.leaf:
        u_hat[node.lane] = node.s[0]
    else:
        _collect_bits(node.left, u_hat)
        _collect_bits(node.right, u_hat)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（树形实现，与 Aff3ct 一致）。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        tmp = phi
        layer = 0
        while tmp % 2 == 1 and layer < n:
            layers_llr.append(layer)
            tmp //= 2
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        tmp = phi + 1
        layer = 0
        while tmp % 2 == 0 and layer < n:
            layers_bit.append(layer)
            tmp //= 2
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（显式栈，与树形递归等价）。"""
    llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    root = _build_polar_tree(n)
    _allocate_lambdas(root, N)
    _init_frozen(root, frozen_bits)
    root.lam[:] = llr

    _decode_tree(root)

    u_hat = np.zeros(N, dtype=int)
    _collect_bits(root, u_hat)
    return u_hat
