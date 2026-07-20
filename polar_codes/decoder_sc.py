"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _b_check(layer, idx):
    """判断节点是 f 还是 g 分支"""
    return (idx // (1 << layer)) % 2


def _s_updater(layer, idx, s):
    """更新部分和 s 数组"""
    if _b_check(layer - 1, idx):
        s[layer, idx] = s[layer - 1, idx]
    else:
        if s[layer - 1, idx] == -1:
            _s_updater(layer - 1, idx, s)
        partner = idx + (1 << (layer - 1))
        if s[layer - 1, partner] == -1:
            _s_updater(layer - 1, partner, s)
        s[layer, idx] = s[layer - 1, idx] ^ s[layer - 1, partner]


def _li(layer, idx, llrs, s):
    """递归计算 LLR"""
    if llrs[layer, idx] != -np.inf:
        return llrs[layer, idx]
    if _b_check(layer, idx) == 0:
        llrs[layer, idx] = f_operation(
            _li(layer + 1, idx, llrs, s),
            _li(layer + 1, idx + (1 << layer), llrs, s)
        )
    else:
        if layer > 0:
            _s_updater(layer, idx - (1 << layer), s)
        llrs[layer, idx] = g_operation(
            _li(layer + 1, idx - (1 << layer), llrs, s),
            _li(layer + 1, idx, llrs, s),
            s[layer, idx - (1 << layer)]
        )
    return llrs[layer, idx]


def sc_decode_layered(llr_ch, frozen_bits):
    """基于分层因子图的 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    llrs = np.full((n + 1, N), -np.inf, dtype=np.float64)
    llrs[n, :] = llr_ch
    s = -np.ones((n + 1, N), dtype=np.int32)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if frozen_bits[phi]:
            s[0, phi] = 0
            llrs[0, phi] = np.inf
            u_hat[phi] = 0
        else:
            llrs[0, phi] = _li(0, phi, llrs, s)
            u_hat[phi] = 1 if llrs[0, phi] < 0 else 0
            s[0, phi] = u_hat[phi]

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, depth, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return u_hat[idx]

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i:i + 1], depth - 1, bit_offset + i)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i:i + 1], depth - 1, bit_offset + half + i)
        return None

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = np.array([1 << i for i in range(n + 1)], dtype=int)

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(np.log2(psi & -psi)))
            psi >>= 1
        if phi % 2 == 0:
            layers_llr.append(n - 1)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 1:
            psi = phi
            layer = 0
            while psi % 2 == 1:
                layers_bit.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（调用高效分层实现）"""
    return sc_decode_layered(llr_ch, frozen_bits)


def verify_sc_decoder():
    """SC 译码无损验证"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = np.zeros((4, 4), dtype=int)
    for i in range(4):
        for j in range(4):
            if (i & j) == j:
                G[i, j] = 1
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)

    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC decode error"

    print("SC decoder verification passed")


if __name__ == "__main__":
    verify_sc_decoder()
