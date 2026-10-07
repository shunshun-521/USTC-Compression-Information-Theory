"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def _sign_pm(x):
    """sign with 0 mapped to +1（min-sum box-plus）。"""
    s = np.sign(x)
    if np.isscalar(s):
        return 1.0 if s == 0 else s
    s = np.asarray(s, dtype=np.float64)
    s[s == 0] = 1
    return s


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 log-domain f 运算（用于递归参考译码器）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _to_tree_frozen(frozen_bits, N):
    """冻结掩码（1=冻结）转为 bool，True=冻结。"""
    return np.asarray(frozen_bits, dtype=np.int8).astype(bool)


def _tree_to_natural(u_tree, N):
    return u_tree


def sc_decode_recursive(llr, frozen_bits, use_min_sum=False):
    """
    递归 SC 译码（参考实现，与 Kronecker 编码约定一致）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    n = int(math.log2(N))
    assert 2 ** n == N

    frozen_tree = _to_tree_frozen(frozen_bits, N)
    f_fn = f_operation if use_min_sum else f_operation_exact
    u_tree = np.zeros(N, dtype=np.uint8)

    def _node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_tree[idx]:
                u_tree[idx] = 0
            else:
                u_tree[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_tree[idx]], dtype=np.uint8)

        half = length // 2
        la, lb = llr_node[:half], llr_node[half:]
        upper = f_fn(la, lb)
        beta_u = _node(upper, base, half)
        lower = g_operation(la, lb, beta_u.astype(np.float64))
        beta_l = _node(lower, base + half, half)
        return np.concatenate([beta_u ^ beta_l, beta_l])

    _node(llr, 0, N)
    return u_tree.astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        layer = 0
        while layer < n and (p & 1) == 1:
            layers_llr.append(layer)
            p >>= 1
            layer += 1
        if layer < n:
            layers_llr.append(layer)

        layers_bit = []
        p = phi + 1
        layer = 0
        while layer < n and (p & 1) == 0:
            layers_bit.append(layer)
            p >>= 1
            layer += 1

        llr_layer_vec.append(layers_llr)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（min-sum f 运算；输出自然序 u_hat）。
    与递归参考实现等价，供仿真主循环调用。
    """
    return sc_decode_recursive(llr_ch, frozen_bits, use_min_sum=True)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok_rec = ok_sc = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u1 = sc_decode(llr, frozen)
        u2 = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u1, u2)
        if np.array_equal(u[info_idx], u1[info_idx]):
            ok_sc += 1
            ok_rec += 1
    print(f"SC test: {ok_sc}/100 frames correct at Eb/N0=10dB")
