"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _frozen_to_indicator(frozen_bits):
    """将 frozen_bits（True=冻结）转为 1=冻结 的整型指示向量。"""
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return frozen_bits.astype(np.int8)
    return frozen_bits.astype(np.int8)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（Sionna/Arıkan 递归树结构，含中间重编码）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    rev = bit_reversal_permutation(N)
    llr = llr[rev]
    frozen_ind = _frozen_to_indicator(frozen_bits)

    def rec(llr_node, f_ind):
        n = len(llr_node)
        if n == 1:
            u = np.array([0 if f_ind[0] == 1 else (0 if llr_node[0] >= 0 else 1)])
            return u, u

        n2 = n // 2
        llr1, llr2 = llr_node[:n2], llr_node[n2:]
        fi1, fi2 = f_ind[:n2], f_ind[n2:]

        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = rec(x_llr1, fi1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = rec(x_llr2, fi2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_int = (u_hat1_up.astype(int) ^ u_hat2_up.astype(int)).astype(int)
        u_hat_up = np.concatenate([u_hat1_up_int, u_hat2_up.astype(int)])
        return u_hat, u_hat_up

    u_hat, _ = rec(llr, frozen_ind)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        i = 0
        while i < n and ((phi >> i) & 1):
            i += 1
        while i < n:
            llr_layers.append(i)
            i += 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        for j in range(n):
            if (phi >> j) & 1:
                bit_layers.append(j)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（显式栈实现，与递归版本等价）。
    信道 LLR 先做比特倒序置换，与蝶形+倒序编码器配套。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = _frozen_to_indicator(frozen_bits)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]

    ret_stack = []
    call_stack = [("enter", llr_ch, frozen_ind)]

    while call_stack:
        kind, *args = call_stack.pop()
        if kind == "enter":
            llr_node, f_ind = args
            n = len(llr_node)
            if n == 1:
                u = np.array([0 if f_ind[0] == 1 else (0 if llr_node[0] >= 0 else 1)])
                ret_stack.append((u, u))
            else:
                n2 = n // 2
                llr1, llr2 = llr_node[:n2], llr_node[n2:]
                fi1, fi2 = f_ind[:n2], f_ind[n2:]
                call_stack.append(("after_left", llr1, llr2, fi1, fi2))
                call_stack.append(("enter", f_operation(llr1, llr2), fi1))
        elif kind == "after_left":
            llr1, llr2, fi1, fi2 = args
            u1, u1up = ret_stack.pop()
            call_stack.append(("merge", u1, u1up))
            call_stack.append(("enter", g_operation(llr1, llr2, u1up), fi2))
        elif kind == "merge":
            u1, u1up = args
            u2, u2up = ret_stack.pop()
            u_hat = np.concatenate([u1, u2])
            u1up_int = (u1up.astype(int) ^ u2up.astype(int)).astype(int)
            u_hat_up = np.concatenate([u1up_int, u2up.astype(int)])
            ret_stack.append((u_hat, u_hat_up))

    return ret_stack.pop()[0]


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0, seed=0):
    """在极低噪声下验证递归与非递归 SC 译码一致且无错（默认 N=64）。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(seed)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_ref = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_ref), "SC recursive vs non-recursive mismatch"
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error at high SNR"

    return True


if __name__ == "__main__":
    verify_sc_decoders()
    print("SC decoder verification passed.")
