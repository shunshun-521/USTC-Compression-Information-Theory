"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.where(La >= 0, 1.0, -1.0)
    s2 = np.where(Lb >= 0, 1.0, -1.0)
    s1 = np.where(La == 0, 1.0, s1)
    s2 = np.where(Lb == 0, 1.0, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _all_computed(arr):
    return not np.any(np.isnan(arr))


def _up(position):
    p0 = position[0] - 1
    span = 2 ** (position[2] - position[0] + 1)
    p1 = int(np.floor(position[1] / span) * span)
    return [p0, p1, position[2], position[3]]


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp


def _prepare_llr(llr_ch, N):
    """编码含比特倒序，信道 LLR 需做相同倒序后再译码"""
    rev = bit_reversal_permutation(N)
    return llr_ch.astype(np.float64)[rev]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（迭代树遍历，O(N log N)）。
    frozen_bits: 1 表示冻结位，0 表示信息位。
    """
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_positions = set(np.where(~frozen_bits)[0])
    frozen_val = 0

    y_llr = _prepare_llr(llr_ch, N)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        base = position[1]
        up_llr = llr_matrix[position[0]][base : base + span]
        up_bit = bit_matrix[position[0]][base : base + span]
        left_llr = llr_matrix[position[0] + 1][base : base + span // 2]
        left_bit = bit_matrix[position[0] + 1][base : base + span // 2]
        right_llr = llr_matrix[position[0] + 1][base + span // 2 : base + span]
        right_bit = bit_matrix[position[0] + 1][base + span // 2 : base + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit_val = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][base : base + span] = up_bit_val.copy()
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                pos = base + 1
                if pos in info_positions:
                    bit = 0 if right_llr[0] > 0 else 1
                else:
                    bit = frozen_val
                bit_matrix[position[0] + 1][base + span // 2 : base + span] = bit
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr_val = g_operation(up_llr[: span // 2], up_llr[span // 2 :], left_bit)
            llr_matrix[position[0] + 1][base + span // 2 : base + span] = right_llr_val
        elif not _all_computed(left_llr):
            left_llr_val = f_operation(up_llr[: span // 2], up_llr[span // 2 :])
            llr_matrix[position[0] + 1][base : base + span // 2] = left_llr_val
        else:
            if position[0] == position[2] - 1:
                pos = base
                if pos in info_positions:
                    bit = 0 if left_llr[0] >= 0 else 1
                else:
                    bit = frozen_val
                bit_matrix[position[0] + 1][base : base + span // 2] = bit
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现，保持接口一致）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算 SCL 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = np.array([(1 << i) - 1 for i in range(n + 1)], dtype=int)

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while (tmp & 1) == 1:
            llr_layers.append(layer)
            layer += 1
            tmp >>= 1
        llr_layer_vec.append(llr_layers)

        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layers = []
            tmp = phi
            layer = 0
            while (tmp & 1) == 1:
                bit_layers.append(layer)
                layer += 1
                tmp >>= 1
            bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.001)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test: {errors}/100 errors (noiseless)")

    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        u_hat = sc_decode(compute_llr(y, sigma), frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test: {errors}/100 errors (Eb/N0=10dB)")
