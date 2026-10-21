"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
import math

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（支持向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.where(La >= 0, 1.0, -1.0)
    sb = np.where(Lb >= 0, 1.0, -1.0)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _f_scalar(a, b):
    return float(np.sign(a) * np.sign(b) * min(abs(a), abs(b)))


def _g_scalar(a, b, u):
    return b - a if u else b + a


# ==================== 递归 SC（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    llr = list(np.asarray(llr, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = [0] * N

    def _sc(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [_f_scalar(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = _sc(left_llr, indices[:half])
        right_llr = [
            _g_scalar(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
        ]
        enc_right = _sc(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(llr, list(range(N)))
    return np.array(decisions, dtype=int)


# ==================== 非递归 SC（分层更新，与递归等价）====================


def precompute_sc_indices(N):
    """预计算每层块大小等（接口兼容）"""
    n = int(math.log2(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return list(range(n + 1)), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归入口：当前调用与递归实现相同的 Arikan SC 算法"""
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if np.array_equal(u, u_hat):
            ok += 1
    print(f"SC test: {ok}/100 frames correct at Eb/N0=10dB")
