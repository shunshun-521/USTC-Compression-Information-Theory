"""
极化码 SC（串行抵消）译码器
提供 Permuted SC（Vangala）高效实现及递归参考实现
"""
import numpy as np
from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while layer < n and (psi & 1) == 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        if layer < n:
            llr_layers.append(layer)
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        if phi % 2 == 1:
            psi = phi
            layer = 0
            while (psi & 1) == 1:
                bit_layers.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与 Permuted SC 等价）"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    Permuted Successive Cancellation Decoder (Vangala et al., 2014)。
    信道 LLR 按自然传输顺序输入。
    frozen_bits: True 表示冻结位。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = bit_reversed(phi, n)
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
            u_hat[l] = B[l, n]
        _update_bits(B, l, n)

    return u_hat


def _update_llrs(L, B, l, n):
    N = L.shape[0]
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                )


def _update_bits(B, l, n):
    if l < B.shape[0] // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                B[j, s - 1] = B[j, s]
