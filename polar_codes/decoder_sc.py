"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，mcba1n SCD 风格）
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
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, bit):
    if bit == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    if bit == 1:
        return l1 - l2
    return np.nan


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    rev = np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)
    llr = llr[rev]

    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=np.int8)
        return np.array([0 if llr[0] >= 0 else 1], dtype=np.int8)

    half = N // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left = sc_decode_recursive(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[:half], llr[half:], u_left)
    u_right = sc_decode_recursive(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi & 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        layer = 0
        while (psi & 1) == 0 and layer < n:
            bit_layers.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（mcba1n SCD 风格，按比特倒序相位译码）。

    若编码端使用了比特倒序置换，传入的信道 LLR 应先做相同倒序对齐。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    # 编码端 polar_encode 含比特倒序，信道 LLR 需倒序对齐到蝶形因子图
    rev = np.array([_bit_reversed(i, n) for i in range(N)], dtype=int)
    llr_internal = llr_ch[rev]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_internal

    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=np.int8)

    for phase in [_bit_reversed(i, n) for i in range(N)]:
        for stage in range(n - _active_llr_level(phase, n), n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(phase, N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = _upper_llr(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = _lower_llr(
                        L[j, stage],
                        L[j - branch_size, stage],
                        B[j - branch_size, stage + 1],
                    )

        if phase in frozen_set:
            u_hat[phase] = 0
            B[phase, n] = 0
        else:
            u_hat[phase] = 0 if L[phase, n] >= 0 else 1
            B[phase, n] = u_hat[phase]

        if phase >= N / 2:
            for stage in range(n, n - _active_bit_level(phase, n), -1):
                block_size = 2 ** stage
                branch_size = block_size // 2
                for j in range(phase, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, stage - 1] = int(B[j, stage]) ^ int(B[j - branch_size, stage])
                        B[j, stage - 1] = B[j, stage]

    return u_hat
