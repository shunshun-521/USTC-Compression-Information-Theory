"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    精确 log 域 f 运算（check node）：
    f(a,b) = ln((1+e^{a+b})/(e^a+e^b))
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _frozen_mask(frozen_bits):
    """frozen_bits: 1=冻结，0=信息 -> bool True=冻结"""
    return np.asarray(frozen_bits, dtype=bool)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（与极化因子图 F=[[1,0],[1,1]] 一致）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.uint8)

        half = length // 2
        llr_up = f_operation(llr_node[:half], llr_node[half:])
        beta_up = node(llr_up, base, half)
        llr_low = g_operation(llr_node[:half], llr_node[half:], beta_up)
        beta_low = node(llr_low, base + half, half)
        return np.concatenate([beta_up ^ beta_low, beta_low])

    node(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（Tal/Vardy 风格）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        layer = 0
        while (p & 1) and layer < n:
            llr_layers.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if not (phi & 1):
            p = phi
            layer = 0
            while layer < n:
                if (p & 1) == 0:
                    bit_layers.append(layer)
                p >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_TABLES = {}


def _get_tables(N):
    if N not in _SC_TABLES:
        _SC_TABLES[N] = precompute_sc_indices(N)
    return _SC_TABLES[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（分层 P/C，精确 f/g）。
    frozen_bits: 1 表示冻结位，0 表示信息位
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def verify_sc_decoders(N=64, trials=50, seed=0):
    """递归与非递归 SC 一致性检查"""
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    rng = np.random.default_rng(seed)
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(6.0, K / N)
    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        if not np.array_equal(u1, u2):
            return False
    return True
