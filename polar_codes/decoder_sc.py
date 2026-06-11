"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（显式栈实现）
"""
import math

import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    u_hat 为左子树返回的部分和（partial sum）
    """
    return (1 - 2 * u_hat) * La + Lb


def _sc_recursive_core(llr, frozen_bits, u_hat):
    """递归 SC 核心：g 运算使用部分和而非源比特。"""

    def recurse(llr_in, offset):
        m = len(llr_in)
        if m == 1:
            idx = offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_in[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=int)

        half = m // 2
        l1, l2 = llr_in[:half], llr_in[half:]
        u_left = recurse(f_operation(l1, l2), offset)
        u_right = recurse(g_operation(l1, l2, u_left), offset + half)
        return np.concatenate([(u_left ^ u_right), u_right])

    recurse(np.asarray(llr, dtype=np.float64), 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(len(llr), dtype=int)
    _sc_recursive_core(llr, frozen_bits, u_hat)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（显式栈，等价于递归实现）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)

    stack = [{"llr": llr_ch, "off": 0, "stage": 0}]
    returns = []

    while stack:
        frame = stack[-1]
        llr = frame["llr"]
        off = frame["off"]
        stage = frame["stage"]
        m = len(llr)

        if m == 1:
            idx = off
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr[0] >= 0 else 1
            returns.append(np.array([u_hat[idx]], dtype=int))
            stack.pop()
            continue

        half = m // 2
        l1, l2 = llr[:half], llr[half:]

        if stage == 0:
            frame["l1"] = l1
            frame["l2"] = l2
            frame["stage"] = 1
            stack.append({"llr": f_operation(l1, l2), "off": off, "stage": 0})
        elif stage == 1:
            u_left = returns.pop()
            frame["u_left"] = u_left
            frame["stage"] = 2
            stack.append(
                {"llr": g_operation(l1, l2, u_left), "off": off + half, "stage": 0}
            )
        else:
            u_right = returns.pop()
            returns.append(np.concatenate([(frame["u_left"] ^ u_right), u_right]))
            stack.pop()

    return u_hat


def precompute_sc_indices(N):
    """
    预计算 SCL 等迭代译码所需的辅助向量（层索引）。
  """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]

    for phi in range(N):
        cur = phi
        for layer in range(n - 1, -1, -1):
            if cur % 2 == 0:
                llr_layer_vec[phi].append(layer)
            cur //= 2

        p = phi
        layer = 0
        while p % 2 == 1:
            bit_layer_vec[phi].append(layer)
            layer += 1
            p >>= 1

    return lambda_offset, llr_layer_vec, bit_layer_vec


def compute_bit_llr(llr_ch, frozen_bits, u_prefix, phi):
    """
    在给定前缀 u_prefix[0:phi] 的条件下，计算比特 phi 的 LLR。
    供 SCL 译码器使用。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_fixed = np.zeros(N, dtype=int)
    u_fixed[:phi] = u_prefix
    target = {"llr": None}

    def recurse(llr_in, offset):
        m = len(llr_in)
        if m == 1:
            idx = offset
            if idx < phi:
                return np.array([u_fixed[idx]], dtype=int)
            if idx == phi:
                target["llr"] = float(llr_in[0])
                return np.array([0], dtype=int)
            if frozen_bits[idx]:
                u_fixed[idx] = 0
            else:
                u_fixed[idx] = 0 if llr_in[0] >= 0 else 1
            return np.array([u_fixed[idx]], dtype=int)

        half = m // 2
        l1, l2 = llr_in[:half], llr_in[half:]
        u_left = recurse(f_operation(l1, l2), offset)
        u_right = recurse(g_operation(l1, l2, u_left), offset + half)
        return np.concatenate([(u_left ^ u_right), u_right])

    recurse(llr_ch, 0)
    return target["llr"] if target["llr"] is not None else 0.0


def path_metric_penalty(llr, bit):
    hard = 0 if llr >= 0 else 1
    return 0.0 if bit == hard else abs(llr)


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 无损验证失败: {errors} 帧错误"
