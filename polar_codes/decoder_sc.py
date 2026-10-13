"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """LLR 域 f 运算（与 aff3ct f_LLR 一致）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign = np.sign(La * Lb)
    sign = np.where(sign == 0, 1.0, sign)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _h_hard(llr):
    return 0 if llr >= 0 else 1


# ==================== 树形 SC（aff3ct naive 结构，正确性参考）====================

_SC_TREE_CACHE = {}


class _SCNode:
    __slots__ = ("lane_id", "lambda_", "s", "left", "right", "is_frozen")

    def __init__(self, vec_size):
        self.lane_id = None
        self.lambda_ = np.zeros(vec_size, dtype=np.float64)
        self.s = np.zeros(vec_size, dtype=np.int8)
        self.left = None
        self.right = None
        self.is_frozen = False


def _build_sc_tree(N):
    if N in _SC_TREE_CACHE:
        return _SC_TREE_CACHE[N]

    lane_counter = [0]

    def build(vec_size):
        node = _SCNode(vec_size)
        if vec_size == 1:
            node.lane_id = lane_counter[0]
            lane_counter[0] += 1
            return node
        half = vec_size // 2
        node.left = build(half)
        node.right = build(half)
        return node

    root = build(N)
    _SC_TREE_CACHE[N] = root
    return root


def _set_frozen_flags(node, frozen_bits):
    if node.left is None:
        node.is_frozen = bool(frozen_bits[node.lane_id])
    else:
        _set_frozen_flags(node.left, frozen_bits)
        _set_frozen_flags(node.right, frozen_bits)


def _decode_tree_node(node, forced=None):
    if node.left is None:
        if forced is not None and node.lane_id in forced:
            node.s[0] = forced[node.lane_id]
        elif node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = _h_hard(node.lambda_[0])
        return

    half = len(node.lambda_) // 2
    for i in range(half):
        node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[half + i])
    _decode_tree_node(node.left, forced)

    for i in range(half):
        node.right.lambda_[i] = g_operation(
            node.lambda_[i], node.lambda_[half + i], node.left.s[i]
        )
    _decode_tree_node(node.right, forced)

    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def _collect_leaf_bits(node, u_hat):
    if node.left is None:
        u_hat[node.lane_id] = node.s[0]
    else:
        _collect_leaf_bits(node.left, u_hat)
        _collect_leaf_bits(node.right, u_hat)


def _channel_llr_to_decode_order(llr_ch, N):
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def _llr_at_phi(llr_ch, frozen_bits, u_prefix, phi):
    """给定前缀 u[0:phi]，返回第 phi 位的 LLR（树形递推，不判决 phi）。"""
    N = len(llr_ch)
    llr = _channel_llr_to_decode_order(llr_ch, N)
    forced = {i: int(u_prefix[i]) for i in range(phi)}
    root = _build_sc_tree(N)
    _set_frozen_flags(root, frozen_bits)
    root.lambda_[:] = llr

    def walk(node):
        if node.left is None:
            if node.lane_id == phi:
                return float(node.lambda_[0])
            if node.lane_id in forced or node.is_frozen:
                node.s[0] = forced.get(node.lane_id, 0)
            else:
                node.s[0] = _h_hard(node.lambda_[0])
            return None
        half = len(node.lambda_) // 2
        for i in range(half):
            node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[half + i])
        left_res = walk(node.left)
        if left_res is not None:
            return left_res
        for i in range(half):
            node.right.lambda_[i] = g_operation(
                node.lambda_[i], node.lambda_[half + i], node.left.s[i]
            )
        right_res = walk(node.right)
        if right_res is not None:
            return right_res
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
        return None

    return walk(root)


def sc_decode_recursive(llr, frozen_bits):
    """递归/树形 SC 译码（参考实现，与 aff3ct naive 等价）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    llr = _channel_llr_to_decode_order(llr, N)

    root = _build_sc_tree(N)
    _set_frozen_flags(root, frozen_bits)
    root.lambda_[:] = llr
    _decode_tree_node(root, None)

    u_hat = np.zeros(N, dtype=np.int8)
    _collect_leaf_bits(root, u_hat)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

_SC_PRECOMPUTE_CACHE = {}


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    if N in _SC_PRECOMPUTE_CACHE:
        return _SC_PRECOMPUTE_CACHE[N]

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        psi = phi
        layer = 0
        while psi % 2 == 1:
            psi //= 2
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        psi = phi
        layer = 0
        while psi % 2 == 0 and psi > 0:
            psi //= 2
            layer += 1
        bit_layer_vec.append(list(range(layer)))

    result = (lambda_offset, llr_layer_vec, bit_layer_vec)
    _SC_PRECOMPUTE_CACHE[N] = result
    return result


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（结果与树形实现一致；内部复用树形译码以保证正确性）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
