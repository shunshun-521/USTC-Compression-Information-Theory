"""
极化码 SC（串行抵消）译码器
非递归实现参考 Permuted SCD（Vangala et al.）/ polarcodes.SCD
"""
import numpy as np
from encoder import _bit_reverse_index


def _sgn(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum f"""
    return _sgn(La) * _sgn(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g：La 为左（上）支路，Lb 为右（下）支路"""
    return (1.0 - 2.0 * u_hat) * La + Lb


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


def _upper_llr(l1, l2):
    return float(f_operation(l1, l2))


def _lower_llr(l1, l2, b):
    b = int(b)
    if b == 0:
        return float(l1 + l2)
    return float(l1 - l2)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=np.int8)

    decode_order = [_bit_reverse_index(i, n) for i in range(N)]

    for l in decode_order:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    btm = L[j, s]
                    top = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = _lower_llr(btm, top, top_bit)

        if l in frozen_set:
            B[l, n] = 0
            u_hat[l] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            B[l, n] = bit
            u_hat[l] = bit

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与蝶形编码配套）"""
    return sc_decode(llr, frozen_bits)
