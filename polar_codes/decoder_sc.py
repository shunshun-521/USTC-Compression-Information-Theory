"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，McBain SCD）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（boxplus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """对数域精确 f 运算"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, bit):
    return l1 + l2 if bit == 0 else l1 - l2


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
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    from encoder import bit_reversed

    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        br_phi = bit_reversed(phi, n)
        start = n - _active_llr_level(br_phi, n)
        llr_layer_vec.append(list(range(start, n)))

        if br_phi < N // 2:
            bit_layer_vec.append([])
        else:
            start_b = n - _active_bit_level(br_phi, n)
            bit_layer_vec.append(list(range(n, start_b, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=np.int32)

    def decode_block(llr_block, layer, bit_pos):
        if len(llr_block) == 1:
            idx = bit_pos
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_block[0] >= 0 else 1
            return

        half = len(llr_block) // 2
        llr_u = f_operation(llr_block[:half], llr_block[half:])
        decode_block(llr_u, layer - 1, bit_pos)
        llr_u_prime = g_operation(
            llr_block[:half], llr_block[half:], u_hat[bit_pos:bit_pos + half]
        )
        decode_block(llr_u_prime, layer - 1, bit_pos + half)

    decode_block(llr, n, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（McBain SCD）。
    L[:,0] = 信道 LLR，按比特倒序相位译码。
    """
    from encoder import bit_reversed

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=np.int32)

    for phase in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(phase, n), n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(phase, N, block):
                if j % block < branch:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = _lower_llr(
                        L[j, s], L[j - branch, s], int(B[j - branch, s + 1])
                    )

        if phase in frozen_set:
            B[phase, n] = 0
            u_hat[phase] = 0
        else:
            bit = 0 if L[phase, n] >= 0 else 1
            B[phase, n] = bit
            u_hat[phase] = bit

        if phase < N // 2:
            continue

        for s in range(n, n - _active_bit_level(phase, n), -1):
            block = 1 << s
            branch = block // 2
            for j in range(phase, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
