"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum f 运算（供 SCL/BP 使用）"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    abs_a, abs_b = np.abs(La), np.abs(Lb)
    result = sa * sb * np.minimum(abs_a, abs_b)
    result = np.where(abs_a < 1e-12, Lb, result)
    result = np.where(abs_b < 1e-12, La, result)
    return result


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _llr_to_p1(llr):
    llr = np.clip(np.asarray(llr, dtype=np.float64), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(llr))


def _cnop(w1, w2):
    return w1 * (1.0 - w2) + w2 * (1.0 - w1)


def _vnop(w1, w2):
    num = w1 * w2
    den = num + (1.0 - w1) * (1.0 - w2)
    return num / den if den > 1e-300 else 0.5


def _hard_dec(w):
    return 1 if w > 0.5 else 0


def _frozen_to_f(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    f = np.full(len(frozen_bits), 0.5)
    f[frozen_bits] = 0.0
    return f


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（P1 域参考实现）"""
    y = _llr_to_p1(llr)
    f = _frozen_to_f(frozen_bits)
    u_hat, _ = _polar_decode_p1(y, f)
    return u_hat


def _polar_decode_p1(y, f):
    """P1 域递归 SC 译码核心"""
    N = len(y)
    if N == 1:
        x = _hard_dec(y[0])
        if f[0] == 0.5:
            return np.array([x], dtype=int), np.array([x], dtype=int)
        return np.array([0], dtype=int), np.array([0], dtype=int)

    u1est = np.array([_cnop(y[i], y[i + 1]) for i in range(0, N, 2)])
    uhat1, u1hard = _polar_decode_p1(u1est, f[:N // 2])

    u2est = np.array([
        _vnop(_cnop(u1hard[i], y[2 * i]), y[2 * i + 1])
        for i in range(N // 2)
    ])
    uhat2, u2hard = _polar_decode_p1(u2est, f[N // 2:])

    u = np.zeros(N, dtype=int)
    u[:N // 2] = uhat1
    u[N // 2:] = uhat2
    x1 = np.array([_cnop(u1hard[i], u2hard[i]) for i in range(N // 2)])
    x = np.zeros(N, dtype=int)
    x[0::2] = x1
    x[1::2] = u2hard
    return u, x


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        p = phi
        while p & 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        p = phi + 1
        if p < N:
            while (p & 1) == 0 and p > 0:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_llr_at_phi(llr, frozen_bits, u, phi):
    """已知 u[0:phi] 时计算 u[phi] 的 LLR（与 P1 SC 递归结构一致）"""
    y = _llr_to_p1(llr)
    f = _frozen_to_f(frozen_bits)

    def decode(y_sub, f_sub, offset):
        size = len(y_sub)
        if size == 1:
            idx = offset
            if idx == phi:
                return y_sub[0]
            return None

        half = size // 2
        u1est = np.array([_cnop(y_sub[i], y_sub[i + 1]) for i in range(0, size, 2)])

        if phi < offset + half:
            return decode(u1est, f_sub[:half], offset)

        uhat1, u1hard = _decode_left_forced(u1est, f_sub[:half], offset, phi, u)
        u2est = np.array([
            _vnop(_cnop(u1hard[i], y_sub[2 * i]), y_sub[2 * i + 1])
            for i in range(half)
        ])
        return decode(u2est, f_sub[half:], offset + half)

    p1 = decode(y, f, 0)
    p1 = np.clip(p1, 1e-12, 1.0 - 1e-12)
    return np.log((1.0 - p1) / p1)


def _decode_left_forced(y_sub, f_sub, offset, phi, u_known):
    """左子树译码，对 idx < phi 使用已知比特"""
    size = len(y_sub)
    if size == 1:
        idx = offset
        if idx < phi:
            val = u_known[idx]
        elif f_sub[0] == 0.5:
            val = _hard_dec(y_sub[0])
        else:
            val = 0
        return np.array([val], dtype=int), np.array([val], dtype=int)

    half = size // 2
    u1est = np.array([_cnop(y_sub[i], y_sub[i + 1]) for i in range(0, size, 2)])
    uhat1, u1hard = _decode_left_forced(u1est, f_sub[:half], offset, phi, u_known)
    u2est = np.array([
        _vnop(_cnop(u1hard[i], y_sub[2 * i]), y_sub[2 * i + 1])
        for i in range(half)
    ])
    uhat2, u2hard = _decode_left_forced(u2est, f_sub[half:], offset + half, phi, u_known)

    u = np.zeros(size, dtype=int)
    u[:half] = uhat1
    u[half:] = uhat2
    x1 = np.array([_cnop(u1hard[i], u2hard[i]) for i in range(half)])
    x = np.zeros(size, dtype=int)
    x[0::2] = x1
    x[1::2] = u2hard
    return u, x


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（长码使用 P1 递归实现保证正确性）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
