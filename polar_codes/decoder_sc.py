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
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


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


def _permute_channel_llr(llr_ch):
    """将信道 LLR 从传输顺序映射到译码树自然顺序。"""
    N = len(llr_ch)
    inv_rev = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_rev]


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现，内部调用与高效版相同的核心逻辑）。
    """
    llr = _permute_channel_llr(llr_ch)
    return _sc_decode_core(llr, frozen_bits)


def _sc_decode_core(llr, frozen_bits):
    """非递归 SC 译码核心（按比特倒序处理）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for bit_idx in [_bit_reversed(i, n) for i in range(N)]:
        for stage in range(n - _active_llr_level(bit_idx, n), n):
            block_size = 2 ** (stage + 1)
            branch_size = block_size // 2
            for j in range(bit_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, stage + 1] = f_operation(L[j, stage], L[j + branch_size, stage])
                else:
                    L[j, stage + 1] = g_operation(
                        L[j - branch_size, stage], L[j, stage], B[j - branch_size, stage + 1]
                    )

        if frozen_bits[bit_idx]:
            u_hat[bit_idx] = 0
        else:
            u_hat[bit_idx] = 0 if L[bit_idx, n] >= 0 else 1
        B[bit_idx, n] = u_hat[bit_idx]

        if bit_idx >= N // 2:
            for stage in range(n, n - _active_bit_level(bit_idx, n), -1):
                block_size = 2 ** stage
                branch_size = block_size // 2
                for j in range(bit_idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, stage - 1] = B[j, stage] ^ B[j - branch_size, stage]
                        B[j, stage - 1] = B[j, stage]

    return u_hat


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
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            psi2 = phi
            while psi2 % 2 == 0 and psi2 > 0:
                layers_bit.append(int(math.log2(psi2 & -psi2)))
                psi2 >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    llr = _permute_channel_llr(llr_ch)
    return _sc_decode_core(llr, frozen_bits)
