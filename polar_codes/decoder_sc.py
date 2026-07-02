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
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    """判断节点是否为 g 分支"""
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    """惰性更新部分和比特"""
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        sibling = idx + (1 << (layer - 1))
        if s[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sibling]


def _compute_llr(layer, idx, llrs, s):
    """惰性递归计算 LLR"""
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        left = _compute_llr(layer + 1, idx, llrs, s)
        right = _compute_llr(layer + 1, idx + (1 << layer), llrs, s)
        llrs[layer, idx] = f_operation(left, right)
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        left_idx = idx - (1 << layer)
        left = _compute_llr(layer + 1, left_idx, llrs, s)
        right = _compute_llr(layer + 1, idx, llrs, s)
        llrs[layer, idx] = g_operation(left, right, s[layer, left_idx])
    return llrs[layer, idx]


def _prepare_llr(llr_ch):
    """将信道 LLR 调整为译码树所需顺序（与编码端比特倒序对应）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    return llr_ch[bit_reversal_permutation(N)]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _prepare_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = np.arange(N, dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        temp = phi
        while temp & 1:
            temp >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))

        layer = 0
        temp = phi
        while (temp >> 1) & 1:
            temp >>= 1
            layer += 1
        bit_layer_vec.append(list(range(layer + 1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（惰性 LLR 计算）。
    """
    llr_ch = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            s[0, phi] = 0
            llrs[0, phi] = np.inf
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _compute_llr(0, phi, llrs, s)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            s[0, phi] = u_hat[phi]

    return u_hat
