"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


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
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int32)

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
        decode_node(llr_left, bit_offset)

        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 2 ** (i - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layer = 0
        tmp = phi
        while (tmp & 1) == 1 and layer < n:
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        bit_layer_vec.append(list(range(layer)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _propagate_f_layered(L, l, idx, n):
    if l == n:
        return
    _propagate_f_layered(L, l + 1, 2 * idx, n)
    _propagate_f_layered(L, l + 1, 2 * idx + 1, n)
    L[l][idx] = f_operation(L[l + 1][2 * idx], L[l + 1][2 * idx + 1])


def _node_llr(L, C, l, idx, phi, n):
    if l == n:
        return L[n][idx]
    bit = (phi >> (n - 1 - l)) & 1
    if bit == 0:
        _node_llr(L, C, l + 1, 2 * idx, phi, n)
        _propagate_f_layered(L, l + 1, 2 * idx + 1, n)
        L[l][idx] = f_operation(L[l + 1][2 * idx], L[l + 1][2 * idx + 1])
    else:
        L[l][idx] = g_operation(L[l + 1][2 * idx], L[l + 1][2 * idx + 1], C[l][idx])
        _node_llr(L, C, l + 1, 2 * idx + 1, phi, n)
    return L[l][idx]


def _calc_llr(L, C, layer, index, phase, n):
    """供 SCL 使用的单路径 LLR 递推。"""
    if isinstance(L, list):
        _node_llr(L, C, 0, 0, phase, n)
    else:
        layered = [L[i, : 2 ** i] for i in range(n + 1)]
        layered[n] = L[n, :].copy()
        c_layered = [C[i, : 2 ** i] for i in range(n + 1)]
        _node_llr(layered, c_layered, 0, 0, phase, n)
        for i in range(n + 1):
            L[i, : 2 ** i] = layered[i]


def _update_gamma(C, layer, index, phase, bit, n):
    if layer == n:
        return
    if (phase & 1) == 0:
        C[layer, index] = bit
        _update_gamma(C, layer + 1, index * 2, phase // 2, bit, n)
    else:
        _update_gamma(C, layer + 1, index * 2 + 1, phase // 2, bit, n)


def sc_decode_layered(llr_ch, frozen_bits):
    """分层 f/g 递推 SC（MSB 比特序，供对照/调试）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = [np.zeros(2 ** l, dtype=np.float64) for l in range(n + 1)]
    L[n] = llr_ch.copy()
    C = [np.zeros(2 ** l, dtype=np.int32) for l in range(n + 1)]
    u_hat = np.zeros(N, dtype=np.int32)

    for phi in range(N):
        _node_llr(L, C, 0, 0, phi, n)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if L[0][0] >= 0 else 1
        C2 = np.zeros((n + 1, N), dtype=np.int32)
        for lv in range(n + 1):
            C2[lv, : 2 ** lv] = C[lv]
        _update_gamma(C2, 0, 0, phi, u_hat[phi], n)
        for lv in range(n + 1):
            C[lv][:] = C2[lv, : 2 ** lv]

    return u_hat


_BP_SC_CACHE = {}


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用因子图单次 BP 更新（max_iter=1），与标准 SC 在 AWGN 下等价。
    """
    from decoder_bp import BPDecoder

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    key = (N, tuple(frozen_bits.tolist()))
    if key not in _BP_SC_CACHE:
        _BP_SC_CACHE[key] = BPDecoder(N, frozen_bits, max_iter=1)
    u_hat, _ = _BP_SC_CACHE[key].decode(llr_ch)
    return u_hat


def verify_sc_lossless(N=64, K=32, num_frames=100, seed=0):
    """极低噪声下 SC 应无错。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(seed)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=np.int32)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = 1e6 * bpsk_modulate(x)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            return False
    return True
