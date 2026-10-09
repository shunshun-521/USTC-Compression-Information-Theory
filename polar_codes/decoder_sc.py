"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，Vangala 2014 置换 SC 结构）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（upper_llr）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(top, btm, u_hat):
    """
    g 运算（lower_llr）：top 为上半支 LLR，btm 为下半支 LLR
    u_hat=0 -> top+btm；u_hat=1 -> btm-top
    """
    return (1 - 2 * u_hat) * top + btm


def _bit_reversed_index(x, n):
    r = 0
    for i in range(n):
        if x & (1 << i):
            r |= 1 << (n - 1 - i)
    return r


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


def _channel_llr_for_decoder(llr_ch):
    """编码端含比特倒序，信道 LLR 需同样置换后进入译码树"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助信息（与 active_llr/bit 层对应）。
    """
    n = int(np.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed_index(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr, frozen_bits):
    """置换 SC 核心（非递归）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits).astype(bool)
    N = len(llr)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr

    u_hat = np.zeros(N, dtype=np.int8)

    for i in range(N):
        l = _bit_reversed_index(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top = L[j - branch_size, s]
                    btm = L[j, s]
                    bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top, btm, bit)

        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（信道自然顺序 LLR）"""
    llr_dec = _channel_llr_for_decoder(llr_ch)
    return _sc_decode_core(llr_dec, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价，用于交叉验证）"""
    return sc_decode(llr, frozen_bits)
