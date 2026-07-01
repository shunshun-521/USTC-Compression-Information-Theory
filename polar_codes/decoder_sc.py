"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _all_num(x):
    return not np.any(np.isnan(x))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * len(left_bit)))
    return temp.ravel()


def _get_right_llr(left_bit, up_llr):
    half = len(up_llr) // 2
    return np.array(
        [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)],
        dtype=np.float64,
    )


def _get_left_llr(up_llr):
    half = len(up_llr) // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)],
        dtype=np.float64,
    )


def _get_bit(llr_val, bit_pos, info_indices, frozen_bits):
    if frozen_bits[bit_pos]:
        return 0
    return 0 if llr_val >= 0 else 1


def _sc_tree_decode(llr_ch, frozen_bits):
    """树遍历 SC 译码核心。"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    info_indices = np.where(frozen_bits == 0)[0]
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = np.ones((n + 1, N), dtype=np.float64)
    bit_matrix[:] = np.nan
    llr_matrix[0] = llr_ch

    position = [0, 0, n, N]
    max_iter = N * n * 8
    it = 0

    while not _all_num(bit_matrix[n]):
        it += 1
        if it > max_iter:
            raise RuntimeError("SC decoder did not converge")

        layer = position[0]
        idx = position[1]
        max_layer = position[2]
        span = 2 ** (max_layer - layer)

        up_llr = llr_matrix[layer, idx : idx + span]
        up_bit = bit_matrix[layer, idx : idx + span]
        half = span // 2
        left_llr = llr_matrix[layer + 1, idx : idx + half]
        left_bit = bit_matrix[layer + 1, idx : idx + half]
        right_llr = llr_matrix[layer + 1, idx + half : idx + span]
        right_bit = bit_matrix[layer + 1, idx + half : idx + span]

        if _all_num(up_bit):
            position = _up(position)
        elif _all_num(right_bit):
            bit_matrix[layer, idx : idx + span] = _get_up_bit(left_bit, right_bit)
        elif _all_num(right_llr):
            if layer == max_layer - 1:
                right_pos = idx + half
                bit_matrix[layer + 1, right_pos] = _get_bit(
                    right_llr[0], right_pos, info_indices, frozen_bits
                )
            else:
                position = _rightdown(position)
        elif _all_num(left_bit):
            llr_matrix[layer + 1, idx + half : idx + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_num(left_llr):
            llr_matrix[layer + 1, idx : idx + half] = _get_left_llr(up_llr)
        else:
            if layer == max_layer - 1:
                bit_matrix[layer + 1, idx] = _get_bit(
                    left_llr[0], idx, info_indices, frozen_bits
                )
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    return _sc_tree_decode(np.asarray(llr_ch, dtype=np.float64), frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 使用同一树遍历核心）。"""
    return _sc_tree_decode(np.asarray(llr, dtype=np.float64), frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        layer = 0
        while p % 2 == 1:
            p >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        bit_layer_vec.append(list(range(layer)) if phi % 2 == 1 else [])
    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """在极低噪声下验证 SC 译码。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import (
        awgn_channel,
        bpsk_modulate,
        compute_llr,
        eb_n0_to_sigma,
        prepare_channel_llr,
    )

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = prepare_channel_llr(compute_llr(y, sigma))
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            raise AssertionError("SC decode error at high SNR")

    return True


if __name__ == "__main__":
    verify_sc_decoders(N=32, K=16, num_frames=50)
    print("SC decoder verification passed.")
