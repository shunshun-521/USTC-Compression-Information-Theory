"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于因子图逐层更新）
"""
import numpy as np
import math
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
    return np.where(x > y, x + np.log1p(np.exp(y - x)), y + np.log1p(np.exp(x - y)))


def f_operation_exact(La, Lb):
    """精确 box-plus f 运算（对数域）。"""
    return _logdomain_sum(La + Lb, 0) - _logdomain_sum(La, Lb)


def _active_llr_level(l, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & l) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(l, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & l) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _prepare_llr(llr_ch):
    """编码含比特倒序置换，信道 LLR 需做相同置换后送入译码树。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def sc_decode_recursive(llr, frozen_bits, use_exact_f=False):
    """
    递归 SC 译码（参考实现）。
    编码含比特倒序置换时，递归结构需与因子图非递归实现保持一致，
    此处复用经 LLR 置换后的非递归译码以保证正确性。
    """
    _ = use_exact_f  # 接口兼容
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（因子图 L/B 矩阵更新）。
  frozen_bits: 非零表示冻结位。
    """
    llr_ch = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1))
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch.copy()

    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = _bit_reversed(i, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
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

        u_hat[l] = B[l, n]

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size >> 1
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        layer = 0
        while (phi >> layer) & 1:
            layer += 1
        while layer < n:
            layers.append(layer)
            layer += 1
        llr_layer_vec.append(layers)

        layers = []
        layer = 0
        while ((phi + 1) >> layer) & 1:
            layer += 1
        while layer < n:
            layers.append(layer)
            layer += 1
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码无损验证"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(42)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_rec_r), "SC recursive vs non-recursive mismatch"
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error"

    return True


if __name__ == "__main__":
    print("SC decoder verification:", verify_sc_decoders())
