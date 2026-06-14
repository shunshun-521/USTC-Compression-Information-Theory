"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
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
    return (1.0 - 2.0 * u_hat) * La + Lb


def _b_check(layer, idx):
    """判断节点 (layer, idx) 是否来自 g 分支。"""
    return (idx // (1 << layer)) % 2 == 1


def _s_updater(layer, idx, bits):
    """递归更新部分比特和。"""
    if layer <= 0:
        return
    if _b_check(layer - 1, idx):
        bits[layer, idx] = bits[layer - 1, idx]
    else:
        if bits[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, bits)
        sibling = idx + (1 << (layer - 1))
        if bits[layer - 1, sibling] == -1:
            _s_updater(layer - 1, sibling, bits)
        bits[layer, idx] = bits[layer - 1, idx] ^ bits[layer - 1, sibling]


def _compute_llr(layer, idx, llrs, bits):
    """惰性计算 (layer, idx) 处 LLR（与参考极化码实现一致）。"""
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        left = _compute_llr(layer + 1, idx, llrs, bits)
        right = _compute_llr(layer + 1, idx + (1 << layer), llrs, bits)
        llrs[layer, idx] = f_operation(left, right)
    else:
        left_idx = idx - (1 << layer)
        if layer > 0:
            _s_updater(layer, left_idx, bits)
        left = _compute_llr(layer + 1, left_idx, llrs, bits)
        right = _compute_llr(layer + 1, idx, llrs, bits)
        llrs[layer, idx] = g_operation(left, right, bits[layer, left_idx])

    return llrs[layer, idx]


def _sc_decode_core(llr_internal, frozen_bits):
    """基于惰性 LLR 的顺序 SC 译码核心。"""
    N = len(llr_internal)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    bits = np.full((n + 1, N), -1, dtype=np.int64)
    llrs[n, :] = llr_internal

    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        llr_phi = _compute_llr(0, phi, llrs, bits)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if llr_phi >= 0 else 1
        bits[0, phi] = u_hat[phi]

    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，内部调用顺序译码核心）。
    """
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（用于文档/扩展）。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * N
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        psi = phi
        while psi % 2 == 1:
            llr_layer_vec[phi].append(int(math.log2(psi & -psi)))
            psi >>= 1
        if psi > 0:
            llr_layer_vec[phi].append(int(math.log2(psi & -psi)))

        if phi % 2 == 0:
            for layer in range(n):
                if (phi >> layer) & 1:
                    bit_layer_vec[phi].append(layer)
        else:
            bit_layer_vec[phi] = [0]

        for layer in range(n):
            if (phi >> layer) & 1:
                lambda_offset[phi] += 1 << layer

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。

    信道 LLR 按编码后的码字顺序输入；内部根据比特倒序置换重排后译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_internal = llr_ch[br]
    return _sc_decode_core(llr_internal, frozen_bits)
