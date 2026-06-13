"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

INF = 1e100


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La * Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    """判断节点属于 g 分支（True）还是 f 分支（False）。"""
    return (idx // (1 << layer)) % 2 == 1


def _s_updater(layer, idx, s):
    """更新部分和比特数组。"""
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] < 0:
            _s_updater(layer - 1, idx, s)
        sibling = idx + (1 << (layer - 1))
        if s[layer - 1, sibling] < 0:
            _s_updater(layer - 1, sibling, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, sibling]


def _li(layer, idx, llrs, s):
    """按需递归计算 LLR。"""
    if llrs[layer, idx] > -INF / 2:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s),
            _li(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        left = idx - (1 << layer)
        if layer > 0:
            _s_updater(layer, left, s)
        llrs[layer, idx] = g_operation(
            _li(layer + 1, left, llrs, s),
            _li(layer + 1, idx, llrs, s),
            s[layer, left],
        )
    return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托给按需计算实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        p = phi
        while p % 2 == 1:
            layers_llr.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            p2 = phi
            while p2 > 0 and p2 % 2 == 0:
                layers_bit.append(int(math.log2(p2 & -p2)))
                p2 >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（自然序信道 LLR）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), -INF, dtype=np.float64)
    llrs[n] = llr_ch
    s = -np.ones((n + 1, N), dtype=np.int8)
    u_hat = np.zeros(N, dtype=np.int8)

    for phi in range(N):
        if frozen_bits[phi]:
            s[0, phi] = 0
            llrs[0, phi] = INF
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _li(0, phi, llrs, s)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            s[0, phi] = u_hat[phi]

    return u_hat
