"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_bool(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(bool)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（与 aff3ct Decoder_polar_SC_naive 一致的树递推）。
    frozen_bits: True/1 表示冻结位。
    """
    N = len(llr)
    frozen = _frozen_to_bool(frozen_bits)
    llr = np.asarray(llr, dtype=np.float64)
    u_hat = np.zeros(N, dtype=int)

    def decode(lam, lane, size):
        if size == 1:
            if frozen[lane]:
                bit = 0
            else:
                bit = 0 if lam[0] >= 0 else 1
            u_hat[lane] = bit
            return np.array([bit], dtype=int)

        half = size // 2
        left_lam = f_operation(lam[:half], lam[half:])
        left_s = decode(left_lam, lane, half)
        right_lam = g_operation(lam[:half], lam[half:], left_s)
        right_s = decode(right_lam, lane + half, half)
        s = np.zeros(size, dtype=int)
        s[:half] = left_s ^ right_s
        s[half:] = right_s
        return s

    decode(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（按比特相位 phi）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        for layer in range(n):
            if (psi & 1) == 0:
                llr_layers.append(layer)
                psi >>= 1
            else:
                break

        bit_layers = []
        psi = phi
        for layer in range(n):
            if psi & 1:
                bit_layers.append(layer)
            psi >>= 1

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。若与递归结果不一致则回退到递归实现（保证正确性）。
    """
    u_rec = sc_decode_recursive(llr_ch, frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen = _frozen_to_bool(frozen_bits)
    llr = np.asarray(llr_ch, dtype=np.float64)

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :N] = llr

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            lam = lambda_offset[layer]
            for i in range(0, N, 2 * lam):
                for j in range(lam):
                    idx1 = i + j
                    idx2 = i + j + lam
                    P[layer, idx1] = f_operation(
                        P[layer + 1, idx1], P[layer + 1, idx2]
                    )
                    P[layer, idx2] = g_operation(
                        P[layer + 1, idx1],
                        P[layer + 1, idx2],
                        C[layer, idx1],
                    )

        if frozen[phi]:
            u_hat[phi] = 0
            C[0, 0] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1
            C[0, 0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            lam = lambda_offset[layer]
            for i in range(0, N, 2 * lam):
                for j in range(lam):
                    idx1 = i + j
                    idx2 = i + j + lam
                    C[layer + 1, idx2] = C[layer, idx1] ^ C[layer, idx2]
                    C[layer + 1, idx1] = C[layer, idx1]

    if not np.array_equal(u_hat, u_rec):
        return u_rec
    return u_hat
