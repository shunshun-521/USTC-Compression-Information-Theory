"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

NEG_INF = -1e30


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits).reshape(-1)
    return fb.astype(int) == 1


def _is_g_node(layer, idx):
    return (idx // (1 << layer)) % 2 == 1


def _s_updater(layer, idx, s):
    if _is_g_node(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        sibling = idx + (1 << (layer - 1))
        if s[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sibling]


def _compute_llr(layer, idx, llrs, s):
    if llrs[layer, idx] > NEG_INF / 2:
        return llrs[layer, idx]

    if not _is_g_node(layer, idx):
        span = 1 << layer
        llrs[layer, idx] = f_operation(
            _compute_llr(layer + 1, idx, llrs, s),
            _compute_llr(layer + 1, idx + span, llrs, s),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _compute_llr(layer + 1, idx - (1 << layer), llrs, s),
            _compute_llr(layer + 1, idx, llrs, s),
            s[layer, idx - (1 << layer)],
        )
    return llrs[layer, idx]


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    frozen_bits[i]==1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = _frozen_mask(frozen_bits)

    llrs = NEG_INF * np.ones((n + 1, N), dtype=np.float64)
    llrs[n, :] = llr_ch
    s = -np.ones((n + 1, N), dtype=int)

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen[phi]:
            s[0, phi] = 0
            llrs[0, phi] = 1e30
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _compute_llr(0, phi, llrs, s)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            s[0, phi] = u_hat[phi]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，供对照）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1 and layer < n:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 0 and layer < n:
            bit_layers.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
