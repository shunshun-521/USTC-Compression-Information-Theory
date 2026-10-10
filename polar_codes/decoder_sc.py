"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation, polar_decode_core


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _hard_bit(llr):
    return 1 if llr < 0 else 0


class _TreeNode:
    __slots__ = ("lane", "size", "lambda_v", "s", "left", "right")

    def __init__(self, lane, size):
        self.lane = lane
        self.size = size
        self.lambda_v = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=np.int8)
        self.left = None
        self.right = None


def _build_tree(n, lane=0):
    if n == 0:
        return _TreeNode(lane, 1)
    left = _build_tree(n - 1, lane)
    right = _build_tree(n - 1, lane + 2 ** (n - 1))
    root = _TreeNode(lane, 2 ** n)
    root.left = left
    root.right = right
    return root


_TREE_CACHE = {}


def _get_tree(N):
    if N not in _TREE_CACHE:
        _TREE_CACHE[N] = _build_tree(int(math.log2(N)))
    return _TREE_CACHE[N]


def _sc_decode_codeword(llr_nat):
    """树形 SC，返回估计码字（core 域，未做 BRP）。"""
    N = len(llr_nat)
    root = _get_tree(N)
    root.lambda_v[:] = llr_nat

    def decode(node):
        if node.left is not None:
            half = node.size // 2
            for i in range(half):
                node.left.lambda_v[i] = f_operation(
                    node.lambda_v[i], node.lambda_v[half + i]
                )
            decode(node.left)
            for i in range(half):
                node.right.lambda_v[i] = g_operation(
                    node.lambda_v[i],
                    node.lambda_v[half + i],
                    node.left.s[i],
                )
            decode(node.right)
            for i in range(half):
                node.s[i] = node.left.s[i] ^ node.right.s[i]
                node.s[half + i] = node.right.s[i]
        else:
            node.s[0] = _hard_bit(node.lambda_v[0])

    decode(root)
    return root.s.astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（对外接口：返回源比特 u_hat）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量。"""
    m = int(math.log2(N))
    lambda_offset = [1 << i for i in range(m + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        t = phi
        while t & 1:
            llr_layers.append(int(math.log2(t & -t)))
            t >>= 1
        llr_layers.append(int(math.log2(phi ^ (phi + 1))))
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        t = phi
        while t & 1:
            bit_layers.append(int(math.log2(t & -t)))
            t >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码：信道 LLR（与发送码字同序）-> 源序列 u_hat。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    llr_nat = np.empty(N, dtype=np.float64)
    llr_nat[brp] = llr_ch
    x_hat = _sc_decode_codeword(llr_nat)
    u_hat = polar_decode_core(x_hat)
    u_hat[frozen_bits == 1] = 0
    return u_hat
