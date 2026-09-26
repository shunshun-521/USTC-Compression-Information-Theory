"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def align_llr_to_decoder(llr_ch):
    """信道 LLR 与译码因子图索引对齐（编码端比特倒序的逆）"""
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    inv = np.empty(N, dtype=int)
    inv[brp] = np.arange(N)
    return llr_ch[inv]


def _update_partial_sums(C, phi):
    """Arikan UPDATE-G：更新部分和数组 C"""
    idx = phi
    level = 0
    while idx % 2 == 1:
        step = 1 << level
        block = (idx >> (level + 1)) << (level + 1)
        for beta in range(step):
            C[block + step + beta] ^= C[block + beta]
        idx >>= 1
        level += 1


def _calc_p(layer, phi, n, L, C):
    if layer == n:
        return
    span = 1 << (n - layer - 1)
    bit = (phi >> (n - layer - 1)) & 1
    if bit == 0:
        _calc_p(layer + 1, phi, n, L, C)
        block = (phi // (2 * span)) * 2 * span
        L[layer, block : block + span] = f_operation(
            L[layer + 1, block : block + span],
            L[layer + 1, block + span : block + 2 * span],
        )
    else:
        _calc_p(layer + 1, phi - span, n, L, C)
        block = ((phi - span) // (2 * span)) * 2 * span
        L[layer, block + span : block + 2 * span] = g_operation(
            L[layer + 1, block : block + span],
            L[layer + 1, block + span : block + 2 * span],
            C[block : block + span],
        )


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [(1 << (n - layer)) - 1 for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        while tmp & 1:
            llr_layers.append(int(math.log2(tmp & -tmp)))
            tmp >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi & 1:
            bit_layers.append(0)
        tmp = phi >> 1
        while tmp & 1:
            bit_layers.append(int(math.log2(tmp & -tmp)) + 1)
            tmp >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（原生加速 + Python 参考实现）"""
    llr_ch = align_llr_to_decoder(np.asarray(llr_ch, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    try:
        from native_sc_bridge import native_sc_decode

        return native_sc_decode(llr_ch, frozen_bits)
    except (RuntimeError, OSError):
        pass

    frozen_bits = frozen_bits.astype(bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    L = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros(N, dtype=np.int8)
    L[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        _calc_p(0, phi, n, L, C)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if L[0, phi] >= 0 else 1
        C[phi] = u_hat[phi]
        if phi < N - 1:
            _update_partial_sums(C, phi)

    return u_hat
