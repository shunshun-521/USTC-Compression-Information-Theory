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
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _prepare_llr(llr_ch, N):
    """信道 LLR 做比特倒序，与编码器输出顺序对齐。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return llr_ch[bit_reversal_permutation(N)]


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


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    frozen_bits: 非零表示冻结位
    """
    N = len(llr)
    llr = _prepare_llr(llr, N)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, fbits):
        n = len(llr_node)
        if n == 1:
            u = 0 if fbits[0] or llr_node[0] >= 0 else 1
            return np.array([u], dtype=int), np.array([u], dtype=int)

        half = n // 2
        u_left, u_left_up = decode_node(
            f_operation(llr_node[:half], llr_node[half:]), fbits[:half]
        )
        u_right, u_right_up = decode_node(
            g_operation(llr_node[:half], llr_node[half:], u_left_up), fbits[half:]
        )
        u_hat = np.concatenate([u_left, u_right])
        u_left_up_new = (u_left_up ^ u_right_up).astype(int)
        u_hat_up = np.concatenate([u_left_up_new, u_right_up])
        return u_hat, u_hat_up

    return decode_node(llr, frozen_bits)[0]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（序贯实现，与递归版本等价）。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    llr = _prepare_llr(llr_ch, N)
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr
    decoded = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        decoded[l] = int(B[l, n])
        _update_bits(B, l, n, N)

    return decoded


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        phi_bin = format(phi, f"0{n}b")
        layers = []
        for bit_pos in range(n):
            if phi_bin[n - 1 - bit_pos] == "0":
                layers.append(n - 1 - bit_pos)
        llr_layer_vec.append(layers)
        bit_layer_vec.append(list(range(n)) if phi % 2 == 0 else list(range(n - 1)))
    lambda_offset = [2 ** layer - 1 for layer in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec
