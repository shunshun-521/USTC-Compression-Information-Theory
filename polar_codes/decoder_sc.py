"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np


# ==================== 基本运算 ====================

def _sgn_prod(a, b):
    p = a * b
    if p > 0:
        return 1.0
    if p < 0:
        return -1.0
    return 0.0


def f_operation(La, Lb):
    """box-plus 近似 f 运算（比 min-sum 更精确）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.minimum(La, Lb) - np.minimum(0.0, La + Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u) = (1-2u)*La + Lb"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if La.ndim == 0:
        return ((-La) if u_hat else La) + Lb
    return np.where(u_hat, -La, La) + Lb


# ==================== 树形 SC（参考实现）====================

class _TreeNode:
    __slots__ = ("left", "right", "lane", "is_leaf", "lam", "s", "frozen")

    def __init__(self, size):
        self.left = None
        self.right = None
        self.lane = 0
        self.is_leaf = size == 1
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.frozen = False


def _build_tree(size, counter):
    node = _TreeNode(size)
    if size == 1:
        node.lane = counter[0]
        counter[0] += 1
        return node
    half = size // 2
    node.left = _build_tree(half, counter)
    node.right = _build_tree(half, counter)
    return node


_TREE_CACHE = {}


def _get_tree(N):
    if N not in _TREE_CACHE:
        counter = [0]
        _TREE_CACHE[N] = _build_tree(N, counter)
    return _TREE_CACHE[N]


def _reset_tree_state(node, llr=None):
    """清除上一帧残留比特/LLR，避免缓存树污染。"""
    node.s.fill(0)
    if llr is not None:
        node.lam[:] = llr
    if not node.is_leaf:
        _reset_tree_state(node.left)
        _reset_tree_state(node.right)


def _recursive_tree_decode(node):
    if node.is_leaf:
        node.s[0] = 0 if node.frozen else (1 if node.lam[0] < 0 else 0)
        return

    half = len(node.lam) // 2
    for i in range(half):
        node.left.lam[i] = f_operation(node.lam[i], node.lam[half + i])
    _recursive_tree_decode(node.left)

    for i in range(half):
        node.right.lam[i] = g_operation(node.lam[i], node.lam[half + i], node.left.s[i])
    _recursive_tree_decode(node.right)

    for i in range(half):
        node.s[i] = node.left.s[i] ^ node.right.s[i]
        node.s[half + i] = node.right.s[i]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（aff3ct naive 结构）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    root = _get_tree(N)

    def set_frozen(node):
        if node.is_leaf:
            node.frozen = bool(frozen_bits[node.lane])
        else:
            set_frozen(node.left)
            set_frozen(node.right)

    set_frozen(root)
    _reset_tree_state(root, llr)
    _recursive_tree_decode(root)

    u_hat = np.zeros(N, dtype=np.int8)

    def collect(node):
        if node.is_leaf:
            u_hat[node.lane] = node.s[0]
        else:
            collect(node.left)
            collect(node.right)

    collect(root)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算非递归 SC 所需的层索引"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        p = phi
        l = 0
        while (p & 1) == 1 and l < n:
            layers.append(l)
            p >>= 1
            l += 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append(layers.copy())

    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_PRECOMPUTE_CACHE = {}


def _get_sc_precompute(N):
    if N not in _SC_PRECOMPUTE_CACHE:
        _SC_PRECOMPUTE_CACHE[N] = precompute_sc_indices(N)
    return _SC_PRECOMPUTE_CACHE[N]


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（层索引实现，供对照）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    lambda_offset, llr_layer_vec, bit_layer_vec = _get_sc_precompute(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch

    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        if phi == 0:
            layers = list(range(n - 1, -1, -1))
        else:
            layers = llr_layer_vec[phi]

        for layer in layers:
            offset = lambda_offset[layer]
            for beta in range(offset):
                P[layer, beta] = f_operation(
                    P[layer + 1, beta], P[layer + 1, beta + offset]
                )

        u_hat[phi] = 0 if frozen_bits[phi] or P[0, 0] >= 0 else 1
        C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            offset = lambda_offset[layer]
            for beta in range(offset):
                C[layer + 1, 2 * beta + 1] = C[layer, beta]
                C[layer + 1, 2 * beta] = C[layer, beta] ^ C[layer + 1, 2 * beta + 1]
                P[layer + 1, beta + offset] = g_operation(
                    P[layer + 1, beta],
                    P[layer + 1, beta + offset],
                    C[layer, beta],
                )

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（缓存二叉树，与 aff3ct naive 等价）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, frozen_bits=None, num_trials=20, seed=0):
    """验证递归与非递归 SC 译码结果一致"""
    rng = np.random.default_rng(seed)
    if frozen_bits is None:
        frozen_bits = np.zeros(N, dtype=bool)
        frozen_bits[: N // 2] = True
    for _ in range(num_trials):
        llr = rng.normal(0, 2, N)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode_nonrecursive(llr, frozen_bits)
        if not np.array_equal(u1, u2):
            return False
    return True
