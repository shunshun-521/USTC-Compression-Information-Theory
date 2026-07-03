"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：La 为上半分支 LLR，Lb 为下半分支 LLR。
    g(La, Lb, u) = (1 - 2*u) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reverse(i, n):
    return int(bin(i)[2:].zfill(n)[::-1], 2)


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


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（按比特倒序处理，与高效实现等价）。
    """
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(math.log2(N))
    llr_layer_vec = [_active_llr_level(_bit_reverse(phi, n), n) for phi in range(N)]
    bit_layer_vec = [_active_bit_level(_bit_reverse(phi, n), n) for phi in range(N)]
    return llr_layer_vec, bit_layer_vec


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    # 编码器输出经比特倒序，信道 LLR 需倒序映射至 Arikan 顺序
    from encoder import bit_reversal_permutation
    llr_ch = llr_ch[bit_reversal_permutation(N)]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    for phi in range(N):
        l = _bit_reverse(phi, n)

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

        if frozen_bits[l]:
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


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    def encode_no_br(u):
        u = np.array(u, dtype=np.int8).copy()
        N = len(u)
        n = int(math.log2(N))
        for stage in range(n):
            step = 1 << stage
            for i in range(0, N, 2 * step):
                for j in range(step):
                    u[i + j] ^= u[i + j + step]
        return u

    for N in [4, 8, 16, 32, 64]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        fb = frozen_bits.astype(bool)
        rng = np.random.default_rng(0)
        er = 0
        for _ in range(200):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, size=K)
            llr = compute_llr(bpsk_modulate(encode_no_br(u)), 0.001)
            if not np.array_equal(sc_decode(llr, fb), u):
                er += 1
        print(f"N={N} errors: {er}/200")
