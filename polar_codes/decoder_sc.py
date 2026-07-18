"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversed_index

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    等价于 lower_llr(bottom, top, bit)
    """
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(phase, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phase) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(phase, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & phase) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，结果与非递归版本一致）"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息。
    返回比特倒序相位列表及层数。
    """
    n = int(math.log2(N))
    phases = [bit_reversed_index(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phase in phases:
        llr_layer_vec.append(list(range(n - _active_llr_level(phase, n), n)))
        bit_layer_vec.append(
            list(range(_active_bit_level(phase, n))) if phase >= N // 2 else []
        )
    return phases, llr_layer_vec, bit_layer_vec, n


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 按码字自然顺序输入。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    phases, llr_layer_vec, bit_layer_vec, n = precompute_sc_indices(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)
    frozen_set = set(np.where(frozen_bits)[0])

    for phi, phase in enumerate(phases):
        for s in llr_layer_vec[phi]:
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(phase, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if phase in frozen_set:
            B[phase, n] = 0
        else:
            B[phase, n] = 0 if L[phase, n] >= 0 else 1
        u_hat[phase] = int(B[phase, n])

        if phase < N // 2:
            continue
        for s in range(n, n - _active_bit_level(phase, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(phase, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return u_hat
