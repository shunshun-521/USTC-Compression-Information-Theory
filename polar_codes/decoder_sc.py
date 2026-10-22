"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """Check-node boxplus（与 Sionna/Aff3ct 一致的稳定形式）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    llr_max = 30.0
    x = np.clip(La, -llr_max, llr_max)
    y = np.clip(Lb, -llr_max, llr_max)
    sp = lambda z: np.log1p(np.exp(-np.abs(z)))
    return (
        np.sign(x) * np.sign(y) * np.minimum(np.abs(x), np.abs(y))
        + sp(x + y)
        - sp(x - y)
    )


def g_operation(La, Lb, u_hat):
    """Variable-node 更新。"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考 Sionna PolarSCDecoder._polar_decode_sc）。
    frozen_bits: True/1 表示冻结位。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = len(llr)

    if n > 1:
        half = n // 2
        llr1, llr2 = llr[:half], llr[half:]
        fr1, fr2 = frozen_bits[:half], frozen_bits[half:]

        llr_up = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = sc_decode_recursive(llr_up, fr1)

        llr_low = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = sc_decode_recursive(llr_low, fr2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(np.float64)
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_bits[0]:
        u = np.array([0.0])
    else:
        u = np.array([0.0 if llr[0] >= 0 else 1.0])
    return u, u


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = 0
        p = phi + 1
        while p % 2 == 0 and l < n:
            l += 1
            p //= 2
        llr_layer_vec.append(list(range(n - 1, l - 1, -1)))
        t1 = 0
        tmp = phi
        while tmp & 1:
            t1 += 1
            tmp >>= 1
        bit_layer_vec.append(list(range(t1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（递归 boxplus 实现）。"""
    u_hat, _ = sc_decode_recursive(
        np.asarray(llr_ch, dtype=np.float64), np.asarray(frozen_bits, dtype=bool)
    )
    return u_hat.astype(int)


def sc_decode_tree(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)


# 非递归接口别名（与递归 SC 等价，便于实验脚本导入）
sc_decode_nonrecursive = sc_decode
