"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def _prepare_channel_llr(llr_ch):
    """将信道 LLR 重排以匹配蝶形编码（无倒序）的因子图顺序。"""
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return np.asarray(llr_ch, dtype=np.float64)[inv_br]


# ==================== 基本运算 ====================

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
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    return int(f"{x:0{n}b}"[::-1], 2)


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


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    """
    llr = _prepare_channel_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    depth = int(np.log2(N)) + 1
    u_hat = np.zeros(N, dtype=int)

    def _f_list(L1, L2):
        return [np.sign(a) * np.sign(b) * min(abs(a), abs(b)) for a, b in zip(L1, L2)]

    def _g_list(L1, L2, bits):
        return [l2 + (1 - 2 * bits[i]) * l1 for i, (l1, l2) in enumerate(zip(L1, L2))]

    def _decode(y, d, node):
        if d == depth - 1:
            if frozen_bits[node]:
                u_hat[node] = 0
            else:
                u_hat[node] = 0 if y[0] >= 0 else 1
            return [u_hat[node]]

        half = len(y) // 2
        left = _f_list(y[:half], y[half:])
        left_bits = _decode(left, d + 1, 2 * node)
        right = _g_list(y[:half], y[half:], left_bits)
        right_bits = _decode(right, d + 1, 2 * node + 1)
        return [(a + b) % 2 for a, b in zip(left_bits, right_bits)] + right_bits

    _decode(list(llr), 0, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        psi = phi
        for layer in range(n):
            if psi % 2 == 0:
                layers.append(layer)
                break
            psi >>= 1
        else:
            layers.append(n - 1)
        llr_layer_vec.append(layers)

        bit_layers = []
        if phi % 2 == 1:
            psi = phi
            for layer in range(n):
                if psi % 2 == 1:
                    bit_layers.append(layer)
                psi >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Vangala 置换 SC）。
    """
    llr_ch = _prepare_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        top_bit,
                    )

        if l in frozen_set:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    return u_hat
