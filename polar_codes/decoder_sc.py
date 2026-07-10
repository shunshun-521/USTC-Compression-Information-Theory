"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    sign(0) 取 +1（标准约定）
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _prepare_frozen(frozen_bits):
    """约定：1/True 表示冻结位，0/False 表示信息位"""
    arr = np.asarray(frozen_bits)
    if np.issubdtype(arr.dtype, np.integer):
        return arr.astype(int) != 0
    return arr.astype(bool)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n - 1, -1, -1))
        else:
            p = phi + 1
            layer = 0
            while p % 2 == 0:
                p //= 2
                layer += 1
            llr_layers = list(range(n - 1, layer - 1, -1))
        llr_layer_vec.append(llr_layers)

        if phi == N - 1:
            bit_layers = []
        else:
            p = phi
            layer = 0
            while p % 2 == 1:
                p //= 2
                layer += 1
            bit_layers = list(range(layer))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _prepare_frozen(frozen_bits)
    N = len(llr_ch)
    llr = llr_ch[bit_reversal_permutation(N)]

    def decode_node(llr_node, frozen_node):
        if len(llr_node) == 1:
            if frozen_node[0]:
                return np.array([0], dtype=int)
            return np.array([0 if llr_node[0] >= 0 else 1], dtype=int)

        half = len(llr_node) // 2
        frozen_l = frozen_node[:half]
        frozen_r = frozen_node[half:]
        llr_l = f_operation(llr_node[:half], llr_node[half:])
        u_l = decode_node(llr_l, frozen_l)
        llr_r = g_operation(llr_node[:half], llr_node[half:], u_l)
        u_r = decode_node(llr_r, frozen_r)
        return np.concatenate([u_l, u_r])

    return decode_node(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 先做比特倒序置换，与编码器 B_N 约定一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = _prepare_frozen(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr = llr_ch[bit_reversal_permutation(N)].copy()
    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :N] = llr

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            sp = lambda_offset[layer]
            for beta in range(0, N, 2 * sp):
                for omega in range(sp):
                    idx = beta + omega
                    P[layer, idx] = f_operation(
                        P[layer + 1, idx], P[layer + 1, idx + sp]
                    )
                    P[layer, idx + sp] = g_operation(
                        P[layer + 1, idx],
                        P[layer + 1, idx + sp],
                        C[layer, idx],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
        C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            sp = lambda_offset[layer]
            for beta in range(0, N, 2 * sp):
                for omega in range(sp):
                    idx = beta + omega
                    C[layer + 1, idx] = (C[layer, idx] + C[layer, idx + sp]) % 2
                    C[layer + 1, idx + sp] = C[layer, idx + sp]

    return u_hat


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma)
        u_sc = sc_decode(llr, frozen)
        u_rec = sc_decode_recursive(llr, frozen)
        if not np.array_equal(u_sc, u_rec):
            errors += 1
        if not np.array_equal(u_sc[info_idx], u[info_idx]):
            errors += 1
    print("SC validation errors:", errors)
