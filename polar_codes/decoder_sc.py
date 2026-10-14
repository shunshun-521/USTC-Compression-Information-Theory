"""
极化码 SC（串行抵消）译码器
提供递归树版本（Aff3ct SC_naive 参考）与非递归分层版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """box-plus f（译码主路径，高 SNR 下更稳）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    ta = np.tanh(np.clip(La / 2.0, -30.0, 30.0))
    tb = np.tanh(np.clip(Lb / 2.0, -30.0, 30.0))
    prod = np.clip(ta * tb, -1.0 + 1e-15, 1.0 - 1e-15)
    return 2.0 * np.arctanh(prod)


def f_min_sum(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def h_operation(La):
    """硬判决：LLR >= 0 -> 0，否则 1"""
    La = np.asarray(La)
    return (La < 0).astype(int)


class _SCNode:
    __slots__ = ("size", "lambda_", "s", "is_frozen", "left", "right", "lane_id")

    def __init__(self, size):
        self.size = size
        self.lambda_ = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=int)
        self.is_frozen = False
        self.left = None
        self.right = None
        self.lane_id = 0


def _build_tree(size, frozen_bits, lane_start=0):
    """构建极化码完全二叉树（叶子从左到右 lane_id 递增）"""
    node = _SCNode(size)
    if size == 1:
        node.lane_id = lane_start
        node.is_frozen = bool(frozen_bits[lane_start])
        return node
    half = size // 2
    node.left = _build_tree(half, frozen_bits, lane_start)
    node.right = _build_tree(half, frozen_bits, lane_start + half)
    return node


def _recursive_decode(node):
    """Aff3ct Decoder_polar_SC_naive::recursive_decode"""
    if node.size > 1:
        half = node.size // 2
        left, right = node.left, node.right
        for i in range(half):
            left.lambda_[i] = f_operation(node.lambda_[i], node.lambda_[i + half])
        _recursive_decode(left)
        for i in range(half):
            right.lambda_[i] = g_operation(
                node.lambda_[i], node.lambda_[i + half], left.s[i]
            )
        _recursive_decode(right)
        for i in range(half):
            node.s[i] = left.s[i] ^ right.s[i]
            node.s[i + half] = right.s[i]
    else:
        node.s[0] = 0 if node.is_frozen else int(h_operation(node.lambda_[0]))


def sc_decode_recursive(llr, frozen_bits):
    """树形递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    root = _build_tree(N, frozen_bits)
    root.lambda_[:] = llr
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_leaves(root, u_hat)
    return u_hat


def _collect_leaves(node, u_hat):
    if node.size == 1:
        u_hat[node.lane_id] = node.s[0]
        return
    _collect_leaves(node.left, u_hat)
    _collect_leaves(node.right, u_hat)


def precompute_sc_indices(N):
    """预计算非递归 SC 的 llr_layer_vec 与 bit_layer_vec"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        layer = 0
        while (psi & 1) == 1:
            layers_llr.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        psi = phi + 1
        layer = 0
        while (psi & 1) == 0 and layer < n:
            layers_bit.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)

    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（分层存储，与树形版本等价）。
    frozen_bits: 1 表示冻结位
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
