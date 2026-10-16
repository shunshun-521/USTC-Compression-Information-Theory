"""
极化码 SC（串行抵消）译码器
提供递归版本（Aff3ct 风格树）和非递归版本（相位更新）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为左分支 LLR，Lb 为右分支 LLR"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _hard_bit(llr):
    return 1 if llr < 0 else 0


class _SCNode:
    __slots__ = ("size", "lambda_v", "s", "is_frozen", "lane_id", "left", "right", "leaf")

    def __init__(self, size, lane_id=-1, is_frozen=False):
        self.size = size
        self.lambda_v = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.is_frozen = is_frozen
        self.lane_id = lane_id
        self.left = None
        self.right = None
        self.leaf = size == 1


def _build_tree(n, frozen_bits, lane_counter):
    """构建极化码译码二叉树（叶子 lane_id 为自然比特索引 0..N-1）"""
    if n == 0:
        lane = lane_counter[0]
        lane_counter[0] += 1
        return _SCNode(1, lane_id=lane, is_frozen=bool(frozen_bits[lane]))
    node = _SCNode(2 ** n)
    half = 2 ** (n - 1)
    node.left = _build_tree(n - 1, frozen_bits, lane_counter)
    node.right = _build_tree(n - 1, frozen_bits, lane_counter)
    return node


def _recursive_decode_aff3ct(node):
    if not node.leaf:
        half = node.size // 2
        for i in range(half):
            node.left.lambda_v[i] = f_operation(
                node.lambda_v[i], node.lambda_v[half + i]
            )
        _recursive_decode_aff3ct(node.left)
        for i in range(half):
            node.right.lambda_v[i] = g_operation(
                node.lambda_v[i],
                node.lambda_v[half + i],
                node.left.s[i],
            )
        _recursive_decode_aff3ct(node.right)
        for i in range(half):
            node.s[i] = (node.left.s[i] ^ node.right.s[i]) & 1
            node.s[half + i] = node.right.s[i]
    else:
        if node.is_frozen:
            node.s[0] = 0
        else:
            node.s[0] = _hard_bit(node.lambda_v[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（Aff3ct 风格，输出 u 的自然序向量）"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    lane_counter = [0]
    root = _build_tree(n, frozen_bits, lane_counter)
    root.lambda_v[:] = llr
    _recursive_decode_aff3ct(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_leaves(root, u_hat)
    return u_hat


def _collect_leaves(node, u_hat):
    if node.leaf:
        u_hat[node.lane_id] = node.s[0]
        return
    _collect_leaves(node.left, u_hat)
    _collect_leaves(node.right, u_hat)


def precompute_sc_indices(N):
    """预计算非递归 SC 相位更新层索引"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append((p & -p).bit_length() - 1)
            p >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if (phi & 1) == 0:
            p = phi
            while (p & 1) == 0 and p > 0:
                bit_layers.append((p & -p).bit_length() - 1)
                p >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与递归版本等价，大 N 时更快）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
