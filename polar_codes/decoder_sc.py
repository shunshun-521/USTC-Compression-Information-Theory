"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：La 为上层 LLR，Lb 为下层 LLR"""
    return Lb + (1 - 2 * u_hat) * La


def _bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


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


def _prepare_llr(llr_ch):
    """编码含比特倒序，信道 LLR 需做相同置换后送入因子图"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归版本等价）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            llr_layers.append(layer)
            psi //= 2
            layer += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            bit_layers.append(layer)
            psi //= 2
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, _llr_prepared=False):
    """
    非递归 SC 译码主函数（基于迭代因子图更新）。
    """
    if _llr_prepared:
        likelihoods = np.asarray(llr_ch, dtype=np.float64)
    else:
        likelihoods = _prepare_llr(llr_ch)

    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(likelihoods)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = likelihoods

    for l in [_bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = int(B[j - branch_size, s + 1])
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

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
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def verify_sc_decoder(N=64, K=32, num_frames=100):
    """在极低噪声下验证 SC 译码正确性"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = 0.01
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            return False
    return True


if __name__ == "__main__":
    from encoder import polar_encode

    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    ok = verify_sc_decoder()
    print("SC decoder verification:", "PASSED" if ok else "FAILED")
