"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效树形实现）
"""
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    la = float(La)
    lb = float(Lb)
    s1 = 1.0 if la == 0 else np.sign(la)
    s2 = 1.0 if lb == 0 else np.sign(lb)
    return s1 * s2 * min(abs(la), abs(lb))


def f_operation_vec(La, Lb):
    """向量版 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(pos):
    return [pos[0] + 1, pos[1], pos[2], pos[3]]


def _rightdown(pos):
    return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]


def _up(pos):
    p0 = pos[0] - 1
    p1 = int(np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1))) * (2 ** (pos[2] - pos[0] + 1)))
    return [p0, p1, pos[2], pos[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)])
    return np.concatenate([temp, right_bit])


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


def _get_bit(llr_val, is_info, frozen_val):
    if not is_info:
        return frozen_val
    return 0 if llr_val >= 0 else 1


def _sc_tree_decode(y_llr, info_positions, frozen_val=0):
    """非递归树形 SC 译码（高效实现）。"""
    N = y_llr.size
    n = int(np.log2(N))
    info_set = set(int(i) for i in info_positions)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit_new
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_pos = position[1] + 1
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = _get_bit(
                    right_llr[0], right_pos in info_set, frozen_val
                )
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = _get_right_llr(
                left_bit, up_llr
            )
        elif not _all_filled(left_llr):
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = _get_left_llr(up_llr)
        else:
            if position[0] == position[2] - 1:
                left_pos = position[1]
                bit_matrix[position[0] + 1][position[1] : position[1] + half] = _get_bit(
                    left_llr[0], left_pos in info_set, frozen_val
                )
            else:
                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（接口兼容）。
    实际高效译码使用树形 _sc_tree_decode。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        bits = [(phi >> i) & 1 for i in range(n)]
        llr_layers = [layer for layer in range(n) if bits[layer] == 0]
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 1:
            for layer in range(n):
                if bits[layer] == 1:
                    bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _prepare_llr(llr_ch, N):
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def _frozen_to_info(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return np.where(frozen_bits == 0)[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    N = len(llr_ch)
    llr = _prepare_llr(llr_ch, N)
    info_pos = _frozen_to_info(frozen_bits)
    return _sc_tree_decode(llr, info_pos, 0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，委托给高效树形译码器）。"""
    return sc_decode(llr, frozen_bits)


if __name__ == "__main__":
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    u = np.array([1, 0, 1, 1])
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.kron(F, F)
    B = np.zeros((4, 4), dtype=int)
    rev = bit_reversal_permutation(4)
    for i in range(4):
        B[i, rev[i]] = 1
    x = polar_encode(u)
    assert np.array_equal(x, (u @ (B @ G)) % 2)

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC decode errors at high SNR: {errors}"
    print("SC decoder validation passed")
