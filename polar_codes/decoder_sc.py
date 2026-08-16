"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
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
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    return Lb + (1 - 2 * u_hat) * La


def _bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
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


def _upper_llr_minsum(l1, l2):
    return f_operation(l1, l2)


def _lower_llr_minsum(l1, l2, b):
    if b == 0:
        return l2 + l1
    return l2 - l1


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        l = 0
        while l < n:
            if (phi >> l) & 1 == 0:
                layers.append(l)
                break
            l += 1
        l += 1
        while l < n:
            layers.append(l)
            l += 1
        llr_layer_vec.append(layers)

        blayers = []
        l = 0
        while l < n and ((phi >> l) & 1):
            blayers.append(l)
            l += 1
        bit_layer_vec.append(blayers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _run_sc_decoder(llr, frozen_bits):
    """SC 译码核心（llr 已做比特倒序）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_arr = np.asarray(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))

    def bit_reversed(i):
        result = 0
        for bit in range(n):
            if i & (1 << bit):
                result |= 1 << (n - 1 - bit)
        return result

    def active_llr_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def active_bit_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def fop(l1, l2):
        return np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2))

    def gop(l1, l2, b):
        return l2 + l1 if b == 0 else l2 - l1

    L = np.zeros((N, n + 1))
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    if frozen_arr.dtype == bool:
        frozen = set(np.where(frozen_arr)[0])
    else:
        frozen = set(np.where(frozen_arr)[0])

    for i in range(N):
        l = bit_reversed(i)
        for s in range(n - active_llr_level(l), n):
            bs = 2 ** (s + 1)
            brs = bs // 2
            for j in range(l, N, bs):
                if j % bs < brs:
                    L[j, s + 1] = fop(L[j, s], L[j + brs, s])
                else:
                    L[j, s + 1] = gop(L[j - brs, s], L[j, s], B[j - brs, s + 1])
        B[l, n] = 0 if l in frozen else (0 if L[l, n] >= 0 else 1)
        if l >= N // 2:
            for s in range(n, n - active_bit_level(l), -1):
                bs = 2 ** s
                brs = bs // 2
                for j in range(l, -1, -bs):
                    if j % bs >= brs:
                        B[j - brs, s - 1] = int(B[j, s]) ^ int(B[j - brs, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（与含比特倒序的编码器配套）"""
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    llr = llr_ch[bit_reversal_permutation(N)]
    return _run_sc_decoder(llr, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from construction import ga_construction

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"encode: {u} -> {x}")
    assert np.array_equal(x, [1, 0, 1, 1])

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    sigma = 0.1
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u_full[info_idx]):
            errors += 1
    print(f"SC test errors: {errors}/100")
    assert errors == 0
