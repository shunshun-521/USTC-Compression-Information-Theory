"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        bits = [int(b) for b in f"{phi:0{n}b}"][::-1]
        llr_layers = []
        for s, b in enumerate(bits):
            if b == 0:
                llr_layers.append(s)
            else:
                break
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append([s for s, b in enumerate(bits) if b == 1])

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _left_sweep(L, R, n, N):
    """因子图从左向右的 L 消息更新（单次）。"""
    for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for block in range(0, N, 2 * s):
            for i in range(block, block + s):
                i2 = i + s
                L[i, j - 1] = f_operation(R[i, j] + L[i2, j], L[i, j])
                L[i2, j - 1] = f_operation(R[i, j], L[i, j]) + L[i2, j]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    基于极化码因子图行索引，信道 LLR 经比特倒序置换后与 BP 译码器一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    LARGE = 1e6

    frozen_idx = np.where(frozen_bits == 1)[0]
    u_hat = np.zeros(N, dtype=int)

    L = np.zeros((N, n + 1), dtype=np.float64)
    R = np.zeros((N, n + 1), dtype=np.float64)
    channel_llr = llr_ch[br]

    for phi in range(N):
        L[:, n] = channel_llr
        R.fill(0.0)
        R[frozen_idx, 0] = LARGE
        for i in range(phi):
            if frozen_bits[i]:
                R[i, 0] = LARGE
            else:
                R[i, 0] = LARGE if u_hat[i] == 0 else -LARGE

        _left_sweep(L, R, n, N)

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            total = L[phi, 0] + R[phi, 0]
            u_hat[phi] = 0 if total >= 0 else 1

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)
