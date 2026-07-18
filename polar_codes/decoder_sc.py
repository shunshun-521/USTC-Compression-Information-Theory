"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    """
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    br = bit_reversal_permutation(N)
    llr_perm = llr[br].astype(np.float64)

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
        for i in range(half):
            decode_node(llr_left[i:i + 1], bit_offset + i)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], bit_offset + half + i)

    decode_node(llr_perm, 0)
    return u_hat


def active_llr_level(index, n):
    """First '1' position in binary expansion of index (Permuted SCD)."""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & index) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(index, n):
    """First '0' position in binary expansion of index (Permuted SCD)."""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & index) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def update_llrs(L, B, index, n, N):
    """Update layered LLR tree L[:, :] up to stage n for bit index."""
    for stage in range(n - active_llr_level(index, n), n):
        block_size = 1 << (stage + 1)
        branch_size = block_size >> 1
        for j in range(index, N, block_size):
            if j % block_size < branch_size:
                L[j, stage + 1] = f_operation(L[j, stage], L[j + branch_size, stage])
            else:
                L[j, stage + 1] = g_operation(
                    L[j - branch_size, stage],
                    L[j, stage],
                    B[j - branch_size, stage + 1],
                )


def update_bits(B, index, n, N):
    """Propagate hard decisions down the layered bit tree."""
    if index < N // 2:
        return
    for stage in range(n, n - active_bit_level(index, n), -1):
        block_size = 1 << stage
        branch_size = block_size >> 1
        for j in range(index, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, stage - 1] = (B[j, stage] + B[j - branch_size, stage]) % 2
                B[j, stage - 1] = B[j, stage]


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（保留供兼容）。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    br = bit_reversal_permutation(N)
    llr_layer_vec = [list(range(n - active_llr_level(br[phi], n), n)) for phi in range(N)]
    bit_layer_vec = [
        list(range(n, n - active_bit_level(br[phi], n), -1)) if br[phi] >= N // 2 else []
        for phi in range(N)
    ]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（分层 L/B 树，Permuted SCD）。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    br = bit_reversal_permutation(N)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = np.asarray(llr_ch, dtype=np.float64)[br]

    u_hat = np.zeros(N, dtype=np.int8)
    for phase in range(N):
        index = br[phase]
        update_llrs(L, B, index, n, N)
        if frozen_bits[index]:
            bit = 0
        else:
            bit = 0 if L[index, n] >= 0 else 1
        B[index, n] = bit
        u_hat[index] = bit
        update_bits(B, index, n, N)

    return u_hat


def verify_sc_lossless(N=64, K=32, num_frames=100, seed=0):
    """在极低噪声下验证 SC 译码无损。"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(seed)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=np.int8)
        info = rng.integers(0, 2, size=K, dtype=np.int8)
        u[info_idx] = info
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info):
            return False
    return True
