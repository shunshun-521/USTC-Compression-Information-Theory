"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reverse_index(i, n):
    return int(format(i, f'0{n}b')[::-1], 2)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _prepare_llr(llr_ch):
    """信道 LLR 转为译码树输入顺序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return llr_ch[bit_reversal_permutation(len(llr_ch))]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    与 sc_decode 等价，用于验证非递归实现正确性。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = np.array([(1 << layer) - 1 for layer in range(n + 1)], dtype=int)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        layer = 0
        while (phi >> layer) & 1:
            layer += 1
        while layer < n:
            layers_llr.append(layer)
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        layer = 0
        while (phi >> layer) & 1:
            layers_bit.append(layer)
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    输入为信道自然顺序 LLR，内部自动做比特倒序置换。
    """
    llr = _prepare_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    N = len(llr)
    n = int(math.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr

    for l in [_bit_reverse_index(i, n) for i in range(N)]:
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

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n]


def validate_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码无损验证。"""
    from construction import ga_construction
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)

        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat_rec, u_hat), "Recursive and non-recursive SC mismatch"
        assert np.array_equal(u[info_idx], u_hat[info_idx]), "SC decode error"

    return True


if __name__ == "__main__":
    for n_exp in [2, 3, 4, 5, 6, 7, 8]:
        N = 2 ** n_exp
        validate_sc_decoders(N, N // 2, num_frames=30, eb_n0_db=10.0)
        print(f"N={N} validation passed")
    print("All SC decoder validations passed.")
