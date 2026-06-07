"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
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


def _reorder_channel_llr(llr_ch, N):
    """编码含比特倒序时，将信道 LLR 重排为译码树自然顺序"""
    brp = bit_reversal_permutation(N)
    return llr_ch[brp]


def _frozen_to_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(frozen_bits)[0])


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    按比特索引递归地执行与分层译码等价的 f/g 更新。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_to_set(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = _reorder_channel_llr(llr_ch, N)

    def decode_bit(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                for j in range(l, -1, -block_size):
                    if j % block_size >= block_size // 2:
                        B[j - block_size // 2, s - 1] = int(B[j, s]) ^ int(
                            B[j - block_size // 2, s]
                        )
                        B[j, s - 1] = B[j, s]

    def recurse(phi):
        if phi >= N:
            return
        l = _bit_reversed(phi, n)
        decode_bit(l)
        recurse(phi + 1)

    recurse(0)
    return B[:, n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        layers = list(range(start, n))
        llr_layer_vec.append(layers)
        bit_start = n - _active_bit_level(l, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（L/B 分层存储）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = _frozen_to_set(frozen_bits)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = _reorder_channel_llr(llr_ch, N)

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N // 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


if __name__ == "__main__":
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print(f"u={u} -> x={x}")

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)

    for name, dec in [("recursive", sc_decode_recursive), ("non-recursive", sc_decode)]:
        errors = 0
        for _ in range(100):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)
            u_hat = dec(llr, frozen_bits)
            if not np.array_equal(u_hat[info_idx], u[info_idx]):
                errors += 1
        print(f"{name}: errors={errors}")
        assert errors == 0, f"{name} SC 译码错误帧数: {errors}"

    print("SC decoder tests passed.")
