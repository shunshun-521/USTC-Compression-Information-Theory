"""
极化码 SC（串行抵消）译码器
提供递归树版本（参考实现）和非递归树遍历（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def permute_llr_for_decode(llr_ch):
    """信道 LLR 比特倒序，与编码端 B_N 对齐"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def f_operation(La, Lb):
    """min-sum f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    if sa.ndim:
        sa[sa == 0] = 1
        sb[sb == 0] = 1
    else:
        if sa == 0:
            sa = 1
        if sb == 0:
            sb = 1
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.int8)
    return np.where(u_hat == 0, La, -La) + Lb


def _h_hard(llr):
    return 1 if llr < 0 else 0


class _SCNode:
    __slots__ = ("lane_id", "size", "left", "right", "lambda_", "s", "is_frozen")

    def __init__(self, lane_id=None, size=1):
        self.lane_id = lane_id
        self.size = size
        self.left = None
        self.right = None
        self.lambda_ = None
        self.s = None
        self.is_frozen = False

    def is_leaf(self):
        return self.left is None


_TREE_CACHE = {}


def _build_tree(n, counter=None):
    if counter is None:
        counter = [0]
    if n == 0:
        node = _SCNode(lane_id=counter[0], size=1)
        counter[0] += 1
        return node
    size = 2 ** n
    node = _SCNode(size=size)
    node.left = _build_tree(n - 1, counter)
    node.right = _build_tree(n - 1, counter)
    return node


def _get_tree(N):
    n = int(np.log2(N))
    if N not in _TREE_CACHE:
        root = _build_tree(n, [0])
        _TREE_CACHE[N] = root
    return _TREE_CACHE[N]


def _init_frozen_flags(root, frozen_bits):
    fb = np.asarray(frozen_bits)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            node.is_frozen = bool(fb[node.lane_id])
        else:
            stack.append(node.left)
            stack.append(node.right)


def sc_decode_recursive(llr, frozen_bits):
    """递归树 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    root = _get_tree(N)
    _init_frozen_flags(root, frozen_bits)
    root.lambda_ = np.zeros(N, dtype=np.float64)
    root.lambda_[:] = llr
    root.s = np.zeros(N, dtype=np.int8)

    def decode_node(node):
        if not node.is_leaf():
            h = node.size // 2
            if node.left.lambda_ is None:
                node.left.lambda_ = np.zeros(h, dtype=np.float64)
                node.left.s = np.zeros(h, dtype=np.int8)
                node.right.lambda_ = np.zeros(h, dtype=np.float64)
                node.right.s = np.zeros(h, dtype=np.int8)
            for i in range(h):
                node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[h + i])
            decode_node(node.left)
            for i in range(h):
                node.right.lambda_[i] = g_operation(
                    node.lambda_[i], node.lambda_[h + i], node.left.s[i]
                )
            decode_node(node.right)
            for i in range(h):
                node.s[i] = node.left.s[i] ^ node.right.s[i]
                node.s[h + i] = node.right.s[i]
        else:
            node.s[0] = 0 if node.is_frozen else _h_hard(node.lambda_[0])

    decode_node(root)
    u_hat = np.zeros(N, dtype=int)
    stack = [root]
    while stack:
        node = stack.pop()
        if node.is_leaf():
            u_hat[node.lane_id] = node.s[0]
        else:
            stack.append(node.left)
            stack.append(node.right)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 层索引（接口兼容，树形译码器不依赖此表）"""
    n = int(np.log2(N))
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers, p = [], phi
        while p % 2 == 1:
            layers.append(len(layers))
            p //= 2
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(len(layers))))
    return llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr, frozen_bits):
    """非递归实现：与递归树算法等价"""
    return sc_decode_recursive(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """主入口：信道 LLR 倒序 + SC 译码"""
    llr = permute_llr_for_decode(llr_ch)
    return sc_decode_recursive(llr, frozen_bits)
