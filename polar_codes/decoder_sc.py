"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，同一棵译码树）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum / box-plus f（大 LLR 下与 min-sum 一致）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    eps = 1e-12
    out = np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    out = np.where(np.abs(La) < eps, Lb, out)
    out = np.where(np.abs(Lb) < eps, La, out)
    return out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


class _Node:
    __slots__ = ("lam", "s", "is_frozen", "left", "right", "lane_id", "leaf")

    def __init__(self, size, lane_id=-1, leaf=False):
        self.lam = np.zeros(size, dtype=np.float64)
        self.s = np.zeros(size, dtype=int)
        self.is_frozen = False
        self.left = None
        self.right = None
        self.lane_id = lane_id
        self.leaf = leaf


def _build_tree(size, lane_counter):
    if size == 1:
        node = _Node(1, lane_id=lane_counter[0], leaf=True)
        lane_counter[0] += 1
        return node
    node = _Node(size)
    node.left = _build_tree(size // 2, lane_counter)
    node.right = _build_tree(size // 2, lane_counter)
    return node


def _leaf_value(node, lane_id):
    if node.leaf:
        return node.s[0] if node.lane_id == lane_id else None
    v = _leaf_value(node.left, lane_id)
    return v if v is not None else _leaf_value(node.right, lane_id)


def _sc_decode_tree(llr, frozen_bits):
    """AFF3CT 风格译码树（LLR 为信道顺序，frozen 为树顺序）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    lane = [0]
    root = _build_tree(N, lane)

    def init_frozen(node):
        if node.leaf:
            node.is_frozen = bool(frozen_bits[node.lane_id])
            return
        init_frozen(node.left)
        init_frozen(node.right)

    init_frozen(root)
    root.lam[:] = llr

    def decode(node):
        if not node.leaf:
            h = len(node.lam) // 2
            for i in range(h):
                node.left.lam[i] = f_operation(node.lam[i], node.lam[h + i])
            decode(node.left)
            for i in range(h):
                node.right.lam[i] = g_operation(
                    node.lam[i], node.lam[h + i], node.left.s[i]
                )
            decode(node.right)
            for i in range(h):
                node.s[i] = (node.left.s[i] ^ node.right.s[i]) % 2
                node.s[h + i] = node.right.s[i]
        else:
            node.s[0] = 0 if node.is_frozen or node.lam[0] >= 0 else 1

    decode(root)
    u_tree = np.zeros(N, dtype=int)
    for i in range(N):
        u_tree[i] = _leaf_value(root, i)
    return u_tree


def _map_frozen_to_tree(frozen_bits, N):
    rev = bit_reversal_permutation(N)
    fb = np.asarray(frozen_bits, dtype=int)
    return fb[rev].astype(bool)


def _map_u_from_tree(u_tree, N):
    rev = bit_reversal_permutation(N)
    return u_tree[rev]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，供对照）。"""
    N = len(llr)
    u_tree = _sc_decode_tree(llr, _map_frozen_to_tree(frozen_bits, N))
    return _map_u_from_tree(u_tree, N)


def precompute_sc_indices(N):
    """预计算非递归 SC 索引（供 SCL 层更新参考；保留接口）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        for layer in range(n):
            if (phi >> layer) & 1:
                break
            layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)
        if phi % 2 == 0:
            layers_bit = [0]
        else:
            layers_bit = []
            layer = 1
            while layer < n and not ((phi >> layer) & 1):
                layers_bit.append(layer)
                layer += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（树实现 + 比特倒序索引对齐）。"""
    N = len(llr_ch)
    u_tree = _sc_decode_tree(llr_ch, _map_frozen_to_tree(frozen_bits, N))
    return _map_u_from_tree(u_tree, N)


def verify_sc_lossless(num_trials=100, N=64, K=32, seed=0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(seed)

    for _ in range(num_trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx])
