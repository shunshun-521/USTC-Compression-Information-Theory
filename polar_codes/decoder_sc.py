"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（树遍历高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1.0, sa)
    sb = np.where(sb == 0, 1.0, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(arr):
    return not np.any(np.isnan(arr))


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
    p1 = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )
    return [position[0] - 1, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(length)]
                    + [right_bit[i] for i in range(length)])
    return temp.reshape(1, 2 * length).flatten()


def _init_matrices(llr_ch, n, N):
    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[llr_matrix == 1] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch
    return llr_matrix, bit_matrix


def _sc_decode_tree(llr_ch, frozen_bits):
    """树遍历 SC 译码，与 G=F^{\\otimes n} 编码配套。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix, bit_matrix = _init_matrices(llr_ch, n, N)
    position = [0, 0, n, N]

    max_iter = N * n * 8
    for _ in range(max_iter):
        if _all_filled(bit_matrix[n]):
            break

        up_llr = llr_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        up_bit = bit_matrix[position[0]][
            position[1] : position[1] + 2 ** (position[2] - position[0])
        ]
        span = 2 ** (position[2] - position[0])
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up = _get_up_bit(left_bit.astype(int), right_bit.astype(int))
            bit_matrix[position[0]][position[1] : position[1] + span] = up
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = position[1] + half
                if frozen_bits[bit_pos]:
                    val = 0
                else:
                    val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr_new = np.array(
                [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
            )
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr_new
        elif not _all_filled(left_llr):
            left_llr_new = np.array(
                [f_operation(up_llr[i], up_llr[i + half]) for i in range(half)]
            )
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                bit_pos = position[1]
                if frozen_bits[bit_pos]:
                    val = 0
                else:
                    val = 0 if left_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][position[1] : position[1] + half] = val
            else:
                position = _leftdown(position)
    else:
        raise RuntimeError("SC decoder did not converge")

    return bit_matrix[n].astype(int)


def _sc_decode_recursive_core(llr, frozen_bits):
    N = len(llr)
    if N == 1:
        if frozen_bits[0]:
            return np.array([0], dtype=int)
        return np.array([0 if llr[0] >= 0 else 1], dtype=int)

    half = N // 2
    llr_left = f_operation(llr[:half], llr[half:])
    u_left = _sc_decode_recursive_core(llr_left, frozen_bits[:half])
    llr_right = g_operation(llr[:half], llr[half:], u_left)
    u_right = _sc_decode_recursive_core(llr_right, frozen_bits[half:])
    return np.concatenate([u_left, u_right])


def sc_decode_recursive(llr, frozen_bits):
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    return _sc_decode_recursive_core(llr, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    return _sc_decode_tree(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers, psi = [], phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers)
        layers_b, psi = [], phi // 2
        while psi % 2 == 1:
            layers_b.append(int(math.log2(psi & -psi)))
            psi >>= 1
        bit_layer_vec.append(layers_b)
    return list(range(n + 1)), llr_layer_vec, bit_layer_vec


sc_decode = sc_decode_nonrecursive


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        sigma = eb_n0_to_sigma(10.0, K / N)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_n = sc_decode_nonrecursive(llr, frozen)
        if not np.array_equal(u[info_idx], u_n[info_idx]):
            errors += 1
    print(f"SC test errors at 10dB: {errors}/100")
