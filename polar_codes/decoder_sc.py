"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _channel_llr_to_decode_order(llr_ch):
    """编码端含比特倒序时，将信道 LLR 重排为译码树顺序（B_N 逆置换）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    out = np.empty_like(llr_ch)
    out[br] = llr_ch
    return out


def _build_polar_tree(N, frozen_bits):
    """构建极化码完全二叉树（叶节点 lane_id 对应 u 索引 0..N-1）。"""
    nodes = []

    def build(size, lane_start):
        node_id = len(nodes)
        nodes.append(
            {
                "size": size,
                "leaf": size == 1,
                "lane": lane_start if size == 1 else -1,
                "frozen": bool(frozen_bits[lane_start]) if size == 1 else False,
                "left": -1,
                "right": -1,
            }
        )
        if size == 1:
            return node_id, lane_start + 1
        left_id, lane = build(size // 2, lane_start)
        right_id, lane = build(size // 2, lane)
        nodes[node_id]["left"] = left_id
        nodes[node_id]["right"] = right_id
        return node_id, lane

    root_id, _ = build(N, 0)
    return nodes, root_id


def _sc_tree_decode(llr_ch, frozen_bits):
    """AFF3CT 风格递归 SC（树节点维护 lambda / 部分和 s 向量）。"""
    llr_ch = _channel_llr_to_decode_order(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    nodes, root_id = _build_polar_tree(N, frozen_bits)

    lambdas = [np.zeros(nodes[i]["size"], dtype=np.float64) for i in range(len(nodes))]
    sums = [np.zeros(nodes[i]["size"], dtype=np.int8) for i in range(len(nodes))]
    lambdas[root_id][:] = llr_ch

    def recursive_decode(node_id):
        node = nodes[node_id]
        if node["leaf"]:
            if node["frozen"]:
                sums[node_id][0] = 0
            else:
                sums[node_id][0] = 0 if lambdas[node_id][0] >= 0 else 1
            return

        size = node["size"]
        half = size // 2
        left_id = node["left"]
        right_id = node["right"]
        parent_l = lambdas[node_id]
        left_l = lambdas[left_id]
        right_l = lambdas[right_id]

        for i in range(half):
            left_l[i] = f_operation(parent_l[i], parent_l[half + i])
        recursive_decode(left_id)

        for i in range(half):
            right_l[i] = g_operation(parent_l[i], parent_l[half + i], sums[left_id][i])
        recursive_decode(right_id)

        parent_s = sums[node_id]
        for i in range(half):
            parent_s[i] = (sums[left_id][i] ^ sums[right_id][i]) & 1
            parent_s[half + i] = sums[right_id][i]

    recursive_decode(root_id)

    u_hat = np.zeros(N, dtype=np.int8)
    for i, node in enumerate(nodes):
        if node["leaf"]:
            u_hat[node["lane"]] = sums[i][0]
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（AFF3CT 树结构，与编码器 G=B_N F^{⊗n} 对齐）。"""
    return _sc_tree_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        while psi % 2 == 1:
            llr_layers.append(len(llr_layers))
            psi >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(list(range(len(llr_layers), n)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（当前复用树形递归核心，接口与非递归一致）。
    """
    return _sc_tree_decode(llr_ch, frozen_bits)
