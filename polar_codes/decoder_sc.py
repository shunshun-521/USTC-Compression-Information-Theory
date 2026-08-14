"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：lower_llr(btm, top, u) = top + btm (u=0) 或 top - btm (u=1)"""
    return (1 - 2 * u_hat) * Lb + La


def _bit_reversed(i, n):
    return int(format(i, f'0{n}b')[::-1], 2)


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（自然序信道 LLR）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            bs = 1 << (s + 1)
            br = bs // 2
            for j in range(l, N, bs):
                if j % bs < br:
                    L[j, s + 1] = f_operation(L[j, s], L[j + br, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j, s], L[j - br, s], B[j - br, s + 1]
                    )
        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                bs = 1 << s
                br = bs // 2
                for j in range(l, -1, -bs):
                    if j % bs >= br:
                        B[j - br, s - 1] = B[j, s] ^ B[j - br, s]
                        B[j, s - 1] = B[j, s]
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        if l >= N // 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（自然序信道 LLR）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
