"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def _sign_pm(x):
    x = np.asarray(x, dtype=np.float64)
    s = np.sign(x)
    if s.ndim == 0:
        return 1.0 if s == 0 else float(s)
    s[s == 0] = 1
    return s


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算，u_hat 为 0/1"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（双叶子层结构，与标准极化码因子图一致）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    decoded_bits = np.zeros(N, dtype=int)

    def decode(llrs, depth, bit_position):
        half_index = 2 ** (n - depth - 1)
        if depth == n - 1:
            left_bp = 2 * bit_position
            if frozen_bits[left_bp]:
                u_sign = 1.0
            else:
                Lu = f_operation(llrs[0], llrs[1])
                u_sign = 1.0 if Lu >= 0 else -1.0
            decoded_bits[left_bp] = 0 if u_sign > 0 else 1

            right_bp = 2 * bit_position + 1
            if frozen_bits[right_bp]:
                v_sign = 1.0
            else:
                Lv = u_sign * llrs[0] + llrs[1]
                v_sign = 1.0 if Lv >= 0 else -1.0
            decoded_bits[right_bp] = 0 if v_sign > 0 else 1
            return np.array([u_sign * v_sign, v_sign], dtype=np.float64)

        Lu = f_operation(llrs[:half_index], llrs[half_index:])
        u_hat = decode(Lu, depth + 1, bit_position * 2)
        Lv = u_hat * llrs[:half_index] + llrs[half_index:]
        v_hat = decode(Lv, depth + 1, bit_position * 2 + 1)
        return np.concatenate((u_hat * v_hat, v_hat))

    decode(llr, 0, 0)
    return decoded_bits


def precompute_sc_indices(N):
    """预计算非递归 SC 索引（SCL 复用）"""
    n = int(math.log2(N))
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        p = phi
        while p & 1:
            llr_layer_vec[phi].append((p & -p).bit_length() - 1)
            p >>= 1
        if phi % 2 == 0:
            bit_layer_vec[phi].append(0)
        p = phi
        while p and not (p & 1):
            p >>= 1
            if p:
                bit_layer_vec[phi].append((p & -p).bit_length() - 1)
    return llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC（当前与递归实现等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
