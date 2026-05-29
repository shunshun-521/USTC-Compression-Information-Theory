"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _f_boxplus(La, Lb):
    """精确 box-plus（递归参考实现）"""
    La = np.clip(La, -30.0, 30.0)
    Lb = np.clip(Lb, -30.0, 30.0)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def _prepare_llr(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    brp = bit_reversal_permutation(len(llr_ch))
    return llr_ch[brp]


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        psi = phi // 2
        layer = 0
        layers_llr = []
        while psi % 2 == 1:
            layers_llr.append(layer)
            layer += 1
            psi //= 2
        layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        if phi % 2 == 0:
            bit_layer_vec.append([layer])
        else:
            bit_layer_vec.append(list(range(layer)))

    return llr_layer_vec, bit_layer_vec


def _update_llr_layers(P, C, layers, N, use_min_sum=True):
    f_fn = f_operation if use_min_sum else _f_boxplus
    for layer in layers:
        stride = 1 << layer
        for block in range(0, N, 2 * stride):
            for j in range(block, block + stride):
                P[layer, j] = f_fn(P[layer + 1, j], P[layer + 1, j + stride])
                P[layer, j + stride] = g_operation(
                    P[layer + 1, j], P[layer + 1, j + stride], C[layer, j]
                )


def _update_bit_layers(C, layers, N):
    for layer in layers:
        stride = 1 << layer
        for block in range(0, N, 2 * stride):
            for j in range(block, block + stride):
                C[layer + 1, j] = np.bitwise_xor(C[layer, j], C[layer, j + stride])
                C[layer + 1, j + stride] = C[layer, j + stride]


def _decode_core(llr_node, frozen_node):
    """递归 SC 译码核心"""
    n = len(llr_node)
    if n == 1:
        if frozen_node[0]:
            u = 0
        else:
            u = 0 if llr_node[0] >= 0 else 1
        return np.array([u], dtype=int), np.array([float(u)])

    half = n // 2
    llr_left = llr_node[:half]
    llr_right = llr_node[half:]

    llr_f = _f_boxplus(llr_left, llr_right)
    u_left, u_left_up = _decode_core(llr_f, frozen_node[:half])

    llr_g = g_operation(llr_left, llr_right, u_left_up)
    u_right, u_right_up = _decode_core(llr_g, frozen_node[half:])

    u_hat = np.concatenate([u_left, u_right])
    u_up = np.concatenate(
        [
            np.bitwise_xor(u_left_up.astype(int), u_right_up.astype(int)).astype(float),
            u_right_up,
        ]
    )
    return u_hat, u_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _prepare_llr(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat, _ = _decode_core(llr, frozen_bits)
    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（显式栈 DFS）"""
    llr = _prepare_llr(llr_ch)
    frozen = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    stack = [{"phase": "enter", "llr": llr, "frozen": frozen, "offset": 0}]
    outputs = []

    while stack:
        frame = stack.pop()
        llr_node = frame["llr"]
        frozen_node = frame["frozen"]
        offset = frame["offset"]
        n = len(llr_node)

        if n == 1:
            if frozen_node[0]:
                u = 0
            else:
                u = 0 if llr_node[0] >= 0 else 1
            u_hat[offset] = u
            outputs.append((np.array([u], dtype=int), np.array([float(u)])))
            continue

        if frame["phase"] == "enter":
            half = n // 2
            llr_left = llr_node[:half]
            llr_right = llr_node[half:]
            llr_f = _f_boxplus(llr_left, llr_right)

            frame["phase"] = "left_done"
            frame["half"] = half
            frame["llr_left"] = llr_left
            frame["llr_right"] = llr_right
            stack.append(frame)
            stack.append(
                {
                    "phase": "enter",
                    "llr": llr_f,
                    "frozen": frozen_node[:half],
                    "offset": offset,
                }
            )
            continue

        if frame["phase"] == "left_done":
            u_left, u_left_up = outputs.pop()
            half = frame["half"]
            llr_g = g_operation(frame["llr_left"], frame["llr_right"], u_left_up)

            frame["phase"] = "right_done"
            frame["u_left"] = u_left
            frame["u_left_up"] = u_left_up
            stack.append(frame)
            stack.append(
                {
                    "phase": "enter",
                    "llr": llr_g,
                    "frozen": frame["frozen"][half:],
                    "offset": offset + half,
                }
            )
            continue

        u_right, u_right_up = outputs.pop()
        u_left = frame["u_left"]
        u_left_up = frame["u_left_up"]
        u_hat[offset : offset + len(u_left)] = u_left
        u_hat[offset + len(u_left) : offset + len(u_left) + len(u_right)] = u_right
        u_up = np.concatenate(
            [
                np.bitwise_xor(u_left_up.astype(int), u_right_up.astype(int)).astype(float),
                u_right_up,
            ]
        )
        outputs.append((np.concatenate([u_left, u_right]), u_up))

    return u_hat


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    u_test = np.array([1, 0, 1, 1])
    x_test = polar_encode(u_test)
    print("Encoder test x =", x_test)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    mismatches = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_rec = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_hat, u_hat_rec):
            mismatches += 1
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC mismatch: {mismatches}/100, errors: {errors}/100 at Eb/N0=10dB")
