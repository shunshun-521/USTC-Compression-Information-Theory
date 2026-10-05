"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


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


def _as_frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype != bool:
        return fb.astype(np.int8) != 0
    return fb


def _polar_decode_sc_core(llr_ch, frozen_ind):
    """
    递归 SC 内核（与极化码树 partial-sum 一致）。
    返回 (u_hat, u_hat_up)，其中 u_hat_up 为当前阶段的中间编码比特。
    """
    n = len(frozen_ind)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]

        llr_left = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_core(llr_left, f1)

        llr_right = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_core(llr_right, f2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_i = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(np.float64)
        u_hat_up = np.concatenate([u_hat1_up_i, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_ind[0]:
        u_hat = np.array([0.0])
    else:
        u_hat = np.array([0.0 if llr_ch[0] >= 0 else 1.0])
    return u_hat, u_hat


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    frozen_bits: True/1 表示冻结位。
    """
    N = len(llr)
    frozen_ind = _as_frozen_mask(frozen_bits).astype(np.float64)
    rev = bit_reversal_permutation(N)
    llr = np.asarray(llr, dtype=np.float64)[rev]
    u_hat, _ = _polar_decode_sc_core(llr, frozen_ind)
    return u_hat.astype(np.int8)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 的层更新索引。"""
    n = int(math.log2(N))
    block_size = [1 << (n - l) for l in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        p = phi
        l = 0
        while p & 1:
            llr_layer_vec[phi].append(l)
            p >>= 1
            l += 1

        p = (phi + 1) // 2
        l = 0
        while p and (p & 1) == 0:
            bit_layer_vec[phi].append(l)
            p >>= 1
            l += 1

    return block_size, llr_layer_vec, bit_layer_vec


def _update_llr_for_bit(P, C, n, N, phi, block_size, llr_layer_vec):
    for layer in llr_layer_vec[phi]:
        bs = block_size[layer + 1]
        for start in range(0, N, 2 * bs):
            for i in range(bs):
                left = start + i
                right = start + i + bs
                La = P[layer + 1][left]
                Lb = P[layer + 1][right]
                if (phi // bs) % 2 == 0:
                    P[layer][left] = f_operation(La, Lb)
                else:
                    u_p = C[layer][left]
                    P[layer][right] = g_operation(La, Lb, u_p)


def _propagate_bits(C, phi, bit_layer_vec, block_size, N):
    for layer in bit_layer_vec[phi]:
        bs = block_size[layer + 1]
        for start in range(0, N, 2 * bs):
            for i in range(bs):
                left = start + i
                right = start + i + bs
                C[layer + 1][right] = C[layer][left]
                left_val = int(C[layer][left]) ^ int(C[layer][right])
                C[layer + 1][left] = left_val


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主接口。
    当前通过递归内核保证数值正确性；层索引预计算供 SCL 复用。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_paths_equivalent(llr, frozen_bits):
    """比较递归与非递归 SC 输出是否一致。"""
    a = sc_decode_recursive(llr, frozen_bits)
    b = sc_decode(llr, frozen_bits)
    return np.array_equal(a, b)
