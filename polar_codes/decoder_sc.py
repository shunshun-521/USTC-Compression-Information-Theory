"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np
import polar_sc_core as sc_core


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return sc_core.f_hf(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return sc_core.g(La, Lb, u_hat)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers, tmp, layer = [], phi, 0
        while layer < n:
            if (tmp & 1) == 0:
                layers.append(layer)
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(layers)
        bit_layers, tmp, layer = [], phi, 0
        while (tmp & 1) == 1 and layer < n:
            bit_layers.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(y_llr, information_pos, frozen_bit=0):
    N = y_llr.size
    n = int(np.log2(N))
    llr_matrix = np.ones((n + 1, N))
    llr_matrix[llr_matrix == 1] = float("nan")
    bit_matrix = llr_matrix.copy()
    llr_matrix[0, :] = y_llr
    position = [0, 0, n, N]

    while sc_core.all_num(bit_matrix[n, :]) == 0:
        up_llr = llr_matrix[position[0], position[1]:position[1] + 2 ** (position[2] - position[0])]
        up_bit = bit_matrix[position[0], position[1]:position[1] + 2 ** (position[2] - position[0])]
        left_llr = llr_matrix[position[0] + 1, position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        left_bit = bit_matrix[position[0] + 1, position[1]:position[1] + 2 ** (position[2] - position[0] - 1)]
        right_llr = llr_matrix[position[0] + 1, position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]
        right_bit = bit_matrix[position[0] + 1, position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])]

        if sc_core.all_num(up_bit):
            position = sc_core.up(position)
        elif sc_core.all_num(right_bit):
            up_bit_val = sc_core.get_up_bit(left_bit, right_bit)
            bit_matrix[position[0], position[1]:position[1] + 2 ** (position[2] - position[0])] = up_bit_val.copy()
        elif sc_core.all_num(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                rb = sc_core.get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos)
                bit_matrix[position[0] + 1, position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = rb
            else:
                position = sc_core.rightdown(position)
        elif sc_core.all_num(left_bit):
            right_llr_val = sc_core.get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1, position[1] + 2 ** (position[2] - position[0] - 1):position[1] + 2 ** (position[2] - position[0])] = right_llr_val
        else:
            if sc_core.all_num(left_llr) == 0:
                left_llr_val = sc_core.get_left_llr(up_llr)
                llr_matrix[position[0] + 1, position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = left_llr_val
            elif position[0] == position[2] - 1:
                left_bit_pos = position[1]
                lb = sc_core.get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos)
                bit_matrix[position[0] + 1, position[1]:position[1] + 2 ** (position[2] - position[0] - 1)] = lb
            else:
                position = sc_core.leftdown(position)

    return bit_matrix[n, :].astype(np.int32)


def sc_decode(llr_ch, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    information_pos = list(np.where(~frozen_bits)[0])
    return _sc_decode_core(np.asarray(llr_ch, dtype=np.float64), information_pos, 0)


def verify_sc_decoder(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(42)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC decode error"


if __name__ == "__main__":
    verify_sc_decoder()
    print("SC verification passed.")
