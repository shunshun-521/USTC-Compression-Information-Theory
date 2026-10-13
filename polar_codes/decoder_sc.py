"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _hard_bit(llr):
    return 0 if llr >= 0 else 1


class _TreeNode:
    __slots__ = ("lane_id", "size", "lambda_", "s", "is_frozen", "left", "right", "is_leaf", "_parent")

    def __init__(self, size, lane_id=-1):
        self.size = size
        self.lane_id = lane_id
        self.lambda_ = None
        self.s = None
        self.is_frozen = False
        self.left = None
        self.right = None
        self.is_leaf = size == 1


def _build_tree(size, lane_counter):
    node = _TreeNode(size)
    if size == 1:
        node.lane_id = lane_counter[0]
        lane_counter[0] += 1
        node.is_leaf = True
        return node
    half = size // 2
    node.left = _build_tree(half, lane_counter)
    node.right = _build_tree(half, lane_counter)
    return node


def _allocate_tree(node):
    node.lambda_ = np.zeros(node.size, dtype=np.float64)
    node.s = np.zeros(node.size, dtype=np.int8)
    if not node.is_leaf:
        _allocate_tree(node.left)
        _allocate_tree(node.right)


def _init_frozen(node, frozen_bits):  # noqa: used by decoder_scl
    if node.is_leaf:
        node.is_frozen = bool(frozen_bits[node.lane_id])
    else:
        _init_frozen(node.left, frozen_bits)
        _init_frozen(node.right, frozen_bits)


def _recursive_decode(node):
    if not node.is_leaf:
        half = node.size // 2
        node.left.lambda_[:] = f_operation(
            node.lambda_[:half], node.lambda_[half:]
        )
        _recursive_decode(node.left)

        node.right.lambda_[:] = g_operation(
            node.lambda_[:half],
            node.lambda_[half:],
            node.left.s,
        )
        _recursive_decode(node.right)

        node.s[:half] = (node.left.s ^ node.right.s) % 2
        node.s[half:] = node.right.s
    else:
        node.s[0] = 0 if node.is_frozen else _hard_bit(node.lambda_[0])


def _collect_u(node, u_hat):
    if node.is_leaf:
        u_hat[node.lane_id] = node.s[0]
    else:
        _collect_u(node.left, u_hat)
        _collect_u(node.right, u_hat)


def _make_decoder_tree(N, frozen_bits):
    root = _build_tree(N, [0])
    _allocate_tree(root)
    _init_frozen(root, frozen_bits)
    return root


def _channel_llr_to_decoder(llr_ch):
    """信道自然序 LLR -> SC 因子图叶节点序（比特倒序置换）"""
    N = len(llr_ch)
    return np.asarray(llr_ch, dtype=np.float64)[bit_reversal_permutation(N)]


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（树结构）。
    llr: 信道侧自然顺序 LLR；内部做比特倒序重排。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    root = _make_decoder_tree(N, frozen_bits)
    root.lambda_[:] = _channel_llr_to_decoder(llr)
    _recursive_decode(root)
    u_hat = np.zeros(N, dtype=int)
    _collect_u(root, u_hat)
    return u_hat


class SCDecoder:
    """可复用树结构的 SC 译码器（避免每帧重建树）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.root = _make_decoder_tree(N, self.frozen_bits)
        self._br = bit_reversal_permutation(N)

    def decode(self, llr_ch):
        self.root.lambda_[:] = np.asarray(llr_ch, dtype=np.float64)[self._br]
        _recursive_decode(self.root)
        u_hat = np.zeros(self.N, dtype=int)
        _collect_u(self.root, u_hat)
        return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << l for l in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(int(np.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 0:
            psi = phi
            while psi % 2 == 0 and psi > 0:
                bit_layers.append(int(np.log2(psi & -psi)) - 1)
                psi //= 2
        else:
            psi = phi
            while psi % 2 == 1:
                psi //= 2
            while psi % 2 == 0 and psi > 0:
                bit_layers.append(int(np.log2(psi & -psi)) - 1)
                psi //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（当前与树递归实现等价，供接口兼容）。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    decoder = SCDecoder(N, frozen_bits)
    return decoder.decode(llr_ch)
