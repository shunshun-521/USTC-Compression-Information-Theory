"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> Lb-La"""
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


def _frozen_indices(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=int)
    return set(np.where(fb.astype(bool))[0])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，自然序比特索引）"""
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    frozen = np.asarray(frozen_bits, dtype=bool)
    decisions = [None] * N

    def _sc(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left = [f_operation(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = _sc(left, indices[:half])
        right = [
            g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = _sc(right, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(list(llr), list(range(N)))
    return np.array(decisions, dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助信息（比特倒序相位顺序）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SCD 结构，min-sum）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_indices(frozen_bits)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_recursive_matches_nonrecursive(N=64, frozen_bits=None, trials=20, seed=0):
    """验证递归与非递归 SC 一致"""
    rng = np.random.default_rng(seed)
    if frozen_bits is None:
        frozen_bits = np.zeros(N, dtype=int)
        frozen_bits[: N // 2] = 1
    for _ in range(trials):
        llr = rng.normal(0, 2.0, size=N)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u1, u2):
            return False
    return True
