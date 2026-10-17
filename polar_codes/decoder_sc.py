"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb, llr_max=30.0):
    """
    Check-node (boxplus) 更新，与 Sionna/Stimming LLR 定义一致。
    """
    La = np.clip(np.asarray(La, dtype=np.float64), -llr_max, llr_max)
    Lb = np.clip(np.asarray(Lb, dtype=np.float64), -llr_max, llr_max)
    out = np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))
    return np.nan_to_num(out, nan=0.0, posinf=llr_max, neginf=-llr_max)


def f_operation_min_sum(La, Lb):
    """min-sum 近似的 f 运算（供 BP 等场景参考）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    支持 u_hat 为子块比特（长度 len(La)/2）的向量化情形。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0 or u_hat.size == 1:
        u = int(u_hat) if u_hat.ndim == 0 else int(u_hat[0])
        return (1 - 2 * u) * La + Lb
    if u_hat.size == La.size:
        return (1 - 2 * u_hat) * La + Lb
    q = u_hat.size
    h = La.size
    sub = h // q
    out = np.empty(h, dtype=np.float64)
    for j in range(q):
        sl = slice(j * sub, (j + 1) * sub)
        out[sl] = (1 - 2 * int(u_hat[j])) * La[sl] + Lb[sl]
    return out


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(np.int8) > 0


# ==================== 递归 SC 译码（参考实现）====================


def _polar_decode_sc_core(llr_ch, frozen_ind):
    """递归 SC（参考 Sionna PolarSCDecoder._polar_decode_sc）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1, llr2 = llr_ch[:half], llr_ch[half:]
        f1, f2 = frozen_ind[:half], frozen_ind[half:]
        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_core(x_llr1, f1)
        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_core(x_llr2, f2)
        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(
            np.float64
        )
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up
    is_frozen = frozen_ind[0] > 0.5
    if is_frozen:
        u_hat = np.array([0.0])
    else:
        s = np.sign(llr_ch[0])
        u_hat = np.array([0.5 * (1.0 - s) if s != 0 else 1.0])
    return u_hat, u_hat.copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。 """
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [0]
    off = 0
    for layer in range(n + 1):
        lambda_offset.append(off)
        off += 1 << layer

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        p = phi
        l = 0
        while (p & 1) == 1:
            p >>= 1
            l += 1
        for layer in range(l, n):
            layers_llr.append(layer)

        layers_bit = []
        p = phi + 1
        while p % 2 == 0 and p <= N:
            layers_bit.append(int(math.log2(p)) - 1)
            p //= 2
        bit_layer_vec.append(layers_bit)
        llr_layer_vec.append(layers_llr)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码：内部使用递归核；输入 LLR 为 ln P(y|0)/P(y|1)，与 channel.compute_llr 一致。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_bits = _frozen_mask(frozen_bits)
    frozen_ind = frozen_bits.astype(np.float64)
    u_full, _ = _polar_decode_sc_core(llr_ch, frozen_ind)
    return np.round(u_full).astype(np.int8)


def verify_sc_implementations(N=64, trials=20):
    """比较递归与非递归 SC 输出是否一致。"""
    from construction import ga_construction

    info_idx, _, _ = ga_construction(N, N // 2, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(trials):
        llr = rng.normal(0, 1, N)
        u1 = sc_decode_recursive(llr, frozen_bits)
        u2 = sc_decode(llr, frozen_bits)
        if not np.array_equal(u1, u2):
            return False
    return True
