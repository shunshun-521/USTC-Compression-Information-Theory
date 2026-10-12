"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _sc_recursive_core(llrs, offset, size, frozen_bits, decisions):
    """递归 SC 内核：g 运算使用左子树返回的编码比特（非直接 u_hat）。"""
    if size == 1:
        i = offset
        if frozen_bits[i]:
            decisions[i] = 0
        else:
            decisions[i] = 0 if llrs[0] >= 0 else 1
        return [decisions[i]]

    half = size // 2
    left_llr = [
        f_operation(llrs[i], llrs[i + half]) for i in range(half)
    ]
    enc_left = _sc_recursive_core(
        left_llr, offset, half, frozen_bits, decisions
    )
    right_llr = [
        g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)
    ]
    enc_right = _sc_recursive_core(
        right_llr, offset + half, half, frozen_bits, decisions
    )
    return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = np.zeros(N, dtype=int)
    _sc_recursive_core(llr.tolist(), 0, N, frozen_bits, decisions)
    return decisions


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码：显式栈模拟递归过程，避免 Python 递归深度限制。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    decisions = np.zeros(N, dtype=int)
    enc_cache = {}

    # 帧栈元素: (offset, size, llrs, stage)
    # stage: 'down_left' | 'down_right' | 'up'
    stack = [(0, N, llr_ch.tolist(), "down_left")]

    while stack:
        offset, size, llrs, stage = stack.pop()

        if stage == "down_left":
            if size == 1:
                i = offset
                if frozen_bits[i]:
                    decisions[i] = 0
                else:
                    decisions[i] = 0 if llrs[0] >= 0 else 1
                enc_cache[(offset, size)] = [decisions[i]]
                continue

            half = size // 2
            left_llr = [
                f_operation(llrs[i], llrs[i + half]) for i in range(half)
            ]
            stack.append((offset, size, llrs, "after_left"))
            stack.append((offset, half, left_llr, "down_left"))
            continue

        if stage == "after_left":
            half = size // 2
            enc_left = enc_cache[(offset, half)]
            right_llr = [
                g_operation(llrs[i], llrs[i + half], enc_left[i])
                for i in range(half)
            ]
            stack.append((offset, size, llrs, "after_right"))
            stack.append((offset + half, half, right_llr, "down_left"))
            continue

        if stage == "after_right":
            half = size // 2
            enc_left = enc_cache[(offset, half)]
            enc_right = enc_cache[(offset + half, half)]
            enc_cache[(offset, size)] = [
                enc_left[i] ^ enc_right[i] for i in range(half)
            ] + enc_right
            continue

    return decisions


def precompute_sc_indices(N):
    """预计算辅助向量（接口保留，供扩展/调试）。"""
    n = int(np.log2(N))
    lambda_offset = [0]
    for i in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + (1 << (i - 1)))
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def verify_sc_lossless(N=64, K=32, num_frames=100, eb_n0_db=10.0, seed=0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(seed)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        # 近似无噪验证（Eb/N0=10dB 时 SC 仍可能出现误帧）
        llr = compute_llr(bpsk_modulate(x), 1e-6)
        u_r = sc_decode_recursive(llr, frozen_bits)
        u_nr = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_r, u_nr):
            raise AssertionError("recursive vs non-recursive mismatch")
        if not np.array_equal(u[info_idx], u_r[info_idx]):
            raise AssertionError("SC decode error")
    return True
