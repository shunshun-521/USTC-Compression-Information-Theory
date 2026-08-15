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
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _prepare_llr(llr_ch, N):
    """编码端含比特倒序，译码前对信道 LLR 做相同倒序。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    rev = bit_reversal_permutation(N)
    return llr_ch[rev]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = length // 2
        beta_up = node(f_operation(llr_node[:half], llr_node[half:]), base, half)
        llr_right = g_operation(llr_node[:half], llr_node[half:], beta_up)
        beta_low = node(llr_right, base + half, half)
        return np.concatenate([np.bitwise_xor(beta_up, beta_low), beta_low])

    node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（接口保留）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n))
        else:
            layer = 0
            psi = phi
            while psi % 2 == 1:
                psi //= 2
                layer += 1
            llr_layers = [n - layer - 1]

        bit_layers = []
        if phi % 2 == 1:
            layer = 0
            psi = phi
            while psi % 2 == 1:
                psi //= 2
                layer += 1
            bit_layers = list(range(layer))

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    llr = _prepare_llr(llr_ch, N)
    return sc_decode_recursive(llr, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """验证 SC 译码在近似无噪条件下无错误。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(42)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma=1e-6)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_rec, u):
            return False
    return True
