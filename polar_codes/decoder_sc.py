"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（逐位更新 P/C，与因子图一致）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """box-plus f；大 LLR 时用 min-sum 饱和"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    large = (np.abs(La) > 30) | (np.abs(Lb) > 30)
    out = np.empty_like(La)
    if np.any(~large):
        sm = ~large
        out[sm] = 2.0 * np.arctanh(
            np.tanh(La[sm] / 2.0) * np.tanh(Lb[sm] / 2.0)
        )
    if np.any(large):
        lg = large
        out[lg] = np.sign(La[lg]) * np.sign(Lb[lg]) * np.minimum(
            np.abs(La[lg]), np.abs(Lb[lg])
        )
    return out


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _reorder_llr(llr):
    br = bit_reversal_permutation(len(llr))
    return np.asarray(llr, dtype=np.float64)[br]


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        pp = phi
        j = 0
        while j < n and (pp & 1):
            pp >>= 1
            j += 1
        llr_layer_vec.append(list(range(j, n)))
        pp = phi + 1
        j = 0
        while j < n and (pp & 1):
            pp >>= 1
            j += 1
        bit_layer_vec.append(list(range(j)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    llr = _reorder_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)

    def decode_node(llr_node, bit_offset):
        n_len = len(llr_node)
        if n_len == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = n_len // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    decode_node(llr, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（显式栈，与 sc_decode_recursive 等价）"""
    llr_root = _reorder_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr_root), dtype=int)
    stack = [("subtree", llr_root, 0)]

    while stack:
        item = stack.pop()
        if item[0] == "subtree":
            _, llr_node, bit_offset = item
            if len(llr_node) == 1:
                idx = bit_offset
                if frozen_bits[idx]:
                    u_hat[idx] = 0
                else:
                    u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            else:
                half = len(llr_node) // 2
                llr_left = f_operation(llr_node[:half], llr_node[half:])
                stack.append(("right", llr_node, bit_offset, llr_left))
                for i in range(half - 1, -1, -1):
                    stack.append(("subtree", llr_left[i : i + 1], bit_offset + i))
        elif item[0] == "right":
            _, llr_node, bit_offset, llr_left = item
            half = len(llr_node) // 2
            u_left = u_hat[bit_offset : bit_offset + half]
            llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
            for i in range(half - 1, -1, -1):
                stack.append(("subtree", llr_right[i : i + 1], bit_offset + half + i))

    return u_hat
