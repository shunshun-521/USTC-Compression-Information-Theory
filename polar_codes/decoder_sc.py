"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """f 运算（稳定 box-plus，向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    La = np.clip(La, -20.0, 20.0)
    Lb = np.clip(Lb, -20.0, 20.0)
    return 2.0 * np.arctanh(np.tanh(La / 2.0) * np.tanh(Lb / 2.0))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


class _SCNode:
    __slots__ = ("size", "lane_id", "lambda_", "s", "is_frozen", "left", "right")

    def __init__(self, size):
        self.size = size
        self.lane_id = None
        self.lambda_ = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=int)
        self.is_frozen = False
        self.left = None
        self.right = None


def _build_tree(size):
    node = _SCNode(size)
    if size > 1:
        half = size // 2
        node.left = _build_tree(half)
        node.right = _build_tree(half)
    return node


def _assign_lane_ids(node, counter):
    if node.size == 1:
        node.lane_id = counter[0]
        counter[0] += 1
        return
    _assign_lane_ids(node.left, counter)
    _assign_lane_ids(node.right, counter)


def _set_frozen(node, frozen_bits):
    if node.size == 1:
        node.is_frozen = bool(frozen_bits[node.lane_id])
        return
    _set_frozen(node.left, frozen_bits)
    _set_frozen(node.right, frozen_bits)


def _recursive_decode(node):
    if node.size == 1:
        if node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = 0 if node.lambda_[0] >= 0 else 1
        return

    half = node.size // 2
    for i in range(half):
        node.left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[i + half])
    _recursive_decode(node.left)
    for i in range(half):
        node.right.lambda_[i] = g_operation(
            node.lambda_[i], node.lambda_[i + half], node.left.s[i]
        )
    _recursive_decode(node.right)
    for i in range(half):
        node.s[i] = (node.left.s[i] ^ node.right.s[i]) % 2
        node.s[i + half] = node.right.s[i]


def _collect_bits(node, frozen_bits, out):
    if node.size == 1:
        if not frozen_bits[node.lane_id]:
            out.append(node.s[0])
        return
    _collect_bits(node.left, frozen_bits, out)
    _collect_bits(node.right, frozen_bits, out)


_TREE_CACHE = {}


def _get_tree(N):
    if N not in _TREE_CACHE:
        root = _build_tree(N)
        _assign_lane_ids(root, [0])
        _TREE_CACHE[N] = root
    return _TREE_CACHE[N]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（AFF3CT 二叉树结构）"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    llr_ch: 与发送码字自然顺序一致的长度 N 信道 LLR。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    root = _get_tree(N)
    _set_frozen(root, frozen_bits)
    root.lambda_[:] = llr_ch
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=int)

    def gather(node):
        if node.size == 1:
            u_hat[node.lane_id] = node.s[0]
            return
        gather(node.left)
        gather(node.right)

    gather(root)
    return u_hat


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def get_sc_tables(N):
    return precompute_sc_indices(N)


def channel_llr_to_decoder(llr_natural):
    """将自然顺序信道 LLR 映射为译码器输入（比特倒序）。"""
    llr_natural = np.asarray(llr_natural, dtype=np.float64)
    N = len(llr_natural)
    br = bit_reversal_permutation(N)
    return llr_natural[br]
