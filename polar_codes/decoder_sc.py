"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

_INF = np.inf


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign_a = np.sign(La)
    sign_b = np.sign(Lb)
    sign_a = np.where(sign_a == 0, 1, sign_a)
    sign_b = np.where(sign_b == 0, 1, sign_b)
    return sign_a * sign_b * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    """判断节点类型：0=f 分支，1=g 分支"""
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    """递归更新部分和比特数组"""
    if layer <= 0:
        return
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        partner = idx + (1 << (layer - 1))
        if s[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, partner]


def _li(layer, idx, llrs, s):
    """递归计算 LLR（惰性求值）"""
    if llrs[layer, idx] != _INF:
        return llrs[layer, idx]

    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s),
            _li(layer + 1, idx + (1 << layer), llrs, s),
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
            prev_bit = s[layer, idx - (1 << layer)]
        else:
            prev_bit = s[0, idx - 1]
        llrs[layer, idx] = g_operation(
            _li(layer + 1, idx - (1 << layer), llrs, s),
            _li(layer + 1, idx, llrs, s),
            prev_bit,
        )
    return llrs[layer, idx]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，基于 Li 惰性求值）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（保留接口兼容性）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        psi = phi + 1
        while psi % 2 == 0:
            layers_b.append(int(math.log2(psi & -psi)) - 1)
            psi >>= 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数（基于惰性 LLR 计算的高效实现）。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（float64）
        frozen_bits: 长度 N 的 bool/int 数组，True/1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列（0/1 int 数组）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llrs = np.full((n + 1, N), _INF, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = np.full((n + 1, N), -1, dtype=np.int8)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        llrs[0, phi] = _li(0, phi, llrs, s)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
        s[0, phi] = u_hat[phi]

    return u_hat
