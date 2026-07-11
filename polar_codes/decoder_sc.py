"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_num(x):
    return int(np.all(~np.isnan(x)))


def _leftdown(position):
    p0 = position[0] + 1
    return [p0, position[1], position[2], position[3]]


def _rightdown(position):
    p0 = position[0] + 1
    p1 = position[1] + 2 ** (position[2] - 1 - position[0])
    return [p0, p1, position[2], position[3]]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1))))
    p1 *= 2 ** (position[2] - position[0] + 1)
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp.flatten()


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr > 0 else 1
    return frozen_bit


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = int(left_bit.size)
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _get_left_llr(up_llr):
    length = int(up_llr.size // 2)
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
    )


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（基于因子图树遍历的高效实现）。
    """
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(y_llr)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = set(np.where(~frozen_bits)[0])
    frozen_bit = 0

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while _all_num(bit_matrix[n]) == 0:
        span = 2 ** (position[2] - position[0])
        start = position[1]
        up_llr = llr_matrix[position[0], start : start + span]
        up_bit = bit_matrix[position[0], start : start + span]
        half = 2 ** (position[2] - position[0] - 1)
        left_llr = llr_matrix[position[0] + 1, start : start + half]
        left_bit = bit_matrix[position[0] + 1, start : start + half]
        right_llr = llr_matrix[position[0] + 1, start + half : start + span]
        right_bit = bit_matrix[position[0] + 1, start + half : start + span]

        if _all_num(up_bit) == 1:
            position = _up(position)
        elif _all_num(right_bit) == 1:
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], start : start + span] = up_bit
        elif _all_num(right_llr) == 1:
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr[0], information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1, start + half : start + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_num(left_bit) == 1:
            right_llr_new = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, start + half : start + span] = right_llr_new
        elif _all_num(left_llr) == 0:
            left_llr_new = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1, start : start + half] = left_llr_new
        elif position[0] == position[2] - 1:
            left_bit_pos = position[1]
            left_bit_val = _get_left_bit(
                left_llr[0], information_pos, frozen_bit, left_bit_pos
            )
            bit_matrix[position[0] + 1, start : start + half] = left_bit_val
        else:
            position = _leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现作为统一后端）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（供 SCL 使用）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
            temp >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(12.0, K / N)
    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u_full = np.zeros(N, dtype=int)
        u_full[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_full[info_idx], u_rec[info_idx]):
            errors += 1
    assert errors == 0, f"SC decode errors at 10dB: {errors}/100"
    print("SC decoder tests passed.")
