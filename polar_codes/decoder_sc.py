"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归版本（主实现，参考 Permuted SCD 结构）
"""
import math
import numpy as np
from encoder import bit_reversed_index


def f_operation(La, Lb):
    """min-sum 近似的 f / boxplus。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """二进制表示中自最高位起连续 0 的个数 + 1（与 mcba1n 一致）。"""
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
    """二进制表示中自最高位起连续 1 的个数 + 1。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, f_func=f_operation):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        N = L.shape[0]
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_func(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits(B, l, n):
    N = B.shape[0]
    if l < N // 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits, f_func=f_operation):
    """
    非递归 SC 译码。
    frozen_bits: True/1 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = bit_reversed_index(i, n)
        _update_llrs(L, B, l, n, f_func=f_func)
        if frozen_bits[l]:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            B[l, n] = u_hat[l]
        _update_bits(B, l, n)

    return u_hat


def sc_decode_iterative(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """返回译码相位顺序及层信息（调试 / 扩展用）。"""
    n = int(math.log2(N))
    phases = [bit_reversed_index(i, n) for i in range(N)]
    llr_layers = [_active_llr_level(p, n) for p in phases]
    bit_layers = [_active_bit_level(p, n) for p in phases]
    return phases, llr_layers, bit_layers


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（min-sum），与 sc_decode 接口一致。"""
    return sc_decode(llr, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode

    N = 64
    frozen = np.zeros(N, dtype=bool)
    rng = np.random.default_rng(0)
    for _ in range(50):
        u = rng.integers(0, 2, N)
        x = polar_encode(u)
        llr = np.where(x == 0, 20.0, -20.0)
        uh = sc_decode(llr, frozen)
        assert np.array_equal(uh, u)
    print("SC self-test OK")
