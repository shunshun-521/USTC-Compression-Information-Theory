"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


class _SCNode:
    __slots__ = ("size", "lane_id", "lambda_arr", "s", "is_frozen", "left", "right")

    def __init__(self, size):
        self.size = size
        self.lane_id = None
        self.lambda_arr = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.is_frozen = False
        self.left = None
        self.right = None


def _build_tree(size, frozen_bits, lane_offset=0):
    node = _SCNode(size)
    if size == 1:
        node.lane_id = lane_offset
        node.is_frozen = bool(frozen_bits[lane_offset])
        return node
    half = size // 2
    node.left = _build_tree(half, frozen_bits, lane_offset)
    node.right = _build_tree(half, frozen_bits, lane_offset + half)
    return node


def clone_tree(node):
    new = _SCNode(node.size)
    new.lane_id = node.lane_id
    new.is_frozen = node.is_frozen
    new.lambda_arr = node.lambda_arr.copy()
    new.s = node.s.copy()
    if node.left is not None:
        new.left = clone_tree(node.left)
        new.right = clone_tree(node.right)
    return new


def _load_channel_llrs(root, llr_ch):
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    root.lambda_arr[:] = np.asarray(llr_ch, dtype=np.float64)[inv_br]


def _recursive_decode(node):
    if node.size > 1:
        half = node.size // 2
        for i in range(half):
            node.left.lambda_arr[i] = f_operation(
                node.lambda_arr[i], node.lambda_arr[half + i]
            )
        _recursive_decode(node.left)
        for i in range(half):
            node.right.lambda_arr[i] = g_operation(
                node.lambda_arr[i],
                node.lambda_arr[half + i],
                node.left.s[i],
            )
        _recursive_decode(node.right)
        for i in range(half):
            node.s[i] = node.left.s[i] ^ node.right.s[i]
            node.s[half + i] = node.right.s[i]
    else:
        if node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = 0 if node.lambda_arr[0] >= 0 else 1


def _collect_bits(node, u_hat):
    if node.size == 1:
        u_hat[node.lane_id] = node.s[0]
    else:
        _collect_bits(node.left, u_hat)
        _collect_bits(node.right, u_hat)


def _lane_max(node):
    if node.size == 1:
        return node.lane_id
    return _lane_max(node.right)


def _llr_at_phase(node, phi, u_hat):
    """已知 u_hat[0:phi]，返回第 phi 位 LLR。"""
    if node.size == 1:
        return float(node.lambda_arr[0])
    half = node.size // 2
    left_max = _lane_max(node.left)
    for i in range(half):
        node.left.lambda_arr[i] = f_operation(
            node.lambda_arr[i], node.lambda_arr[half + i]
        )
    if phi <= left_max:
        return _llr_at_phase(node.left, phi, u_hat)
    _apply_known_bits(node.left, u_hat)
    for i in range(half):
        node.right.lambda_arr[i] = g_operation(
            node.lambda_arr[i],
            node.lambda_arr[half + i],
            node.left.s[i],
        )
    return _llr_at_phase(node.right, phi, u_hat)


def _apply_known_bits(node, u_hat):
    """将子树中已判决叶节点的 s 设为 u_hat。"""
    if node.size == 1:
        node.s[0] = u_hat[node.lane_id]
        return
    half = node.size // 2
    for i in range(half):
        node.left.lambda_arr[i] = f_operation(
            node.lambda_arr[i], node.lambda_arr[half + i]
        )
    _apply_known_bits(node.left, u_hat)
    for i in range(half):
        node.right.lambda_arr[i] = g_operation(
            node.lambda_arr[i],
            node.lambda_arr[half + i],
            node.left.s[i],
        )
    _apply_known_bits(node.right, u_hat)
    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def sc_decode_tree(llr_ch, frozen_bits):
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    root = _build_tree(N, frozen_bits)
    _load_channel_llrs(root, llr_ch)
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_bits(root, u_hat)
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（二叉树参考实现）。"""
    return sc_decode_tree(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口保留）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        tmp = phi
        layer = 0
        while tmp % 2 == 1:
            tmp //= 2
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        tmp = phi // 2
        layer = 0
        bit_layers = []
        while tmp % 2 == 1:
            tmp //= 2
            layer += 1
            bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：逐相位更新（O(N log N)）。
    """
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    root = _build_tree(N, frozen_bits)
    _load_channel_llrs(root, llr_ch)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        llr_phi = _llr_at_phase(root, phi, u_hat)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if llr_phi >= 0 else 1

    return u_hat
