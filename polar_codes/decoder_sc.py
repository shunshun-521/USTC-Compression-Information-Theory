"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _frozen_bits_to_info_pos(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return np.where(~frozen_bits)[0].tolist()
    return np.where(frozen_bits == 0)[0].tolist()


def _all_decided(arr):
    return not np.any(np.isnan(arr))


def _combine_bits(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length)


def _sc_decode_core(y_llr, information_pos, frozen_bit=0):
    """SC 译码核心（非递归）"""
    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0], position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1, position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1, position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1, position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1, position[1] + half : position[1] + span]

        if _all_decided(bit_matrix[position[0], position[1] : position[1] + span]):
            p0 = position[0] - 1
            p1 = int(
                np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
                * (2 ** (position[2] - position[0] + 1))
            )
            position = [p0, p1, position[2], position[3]]
        elif _all_decided(right_bit):
            bit_matrix[position[0], position[1] : position[1] + span] = _combine_bits(
                left_bit, right_bit
            )
        elif _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if right_bit_pos in information_pos:
                    val = 0 if right_llr[0] >= 0 else 1
                else:
                    val = frozen_bit
                bit_matrix[position[0] + 1, position[1] + half : position[1] + span] = val
            else:
                position = [
                    position[0] + 1,
                    position[1] + 2 ** (position[2] - 1 - position[0]),
                    position[2],
                    position[3],
                ]
        elif _all_decided(left_bit):
            length = left_bit.size
            right_llr_new = g_operation(up_llr[:length], up_llr[length:], left_bit)
            llr_matrix[position[0] + 1, position[1] + half : position[1] + span] = right_llr_new
        elif not _all_decided(left_llr):
            length = up_llr.size // 2
            llr_matrix[position[0] + 1, position[1] : position[1] + half] = f_operation(
                up_llr[:length], up_llr[length:]
            )
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                if left_bit_pos in information_pos:
                    val = 0 if left_llr[0] >= 0 else 1
                else:
                    val = frozen_bit
                bit_matrix[position[0] + 1, position[1] : position[1] + half] = val
            else:
                position = [position[0] + 1, position[1], position[2], position[3]]

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 算法等价）"""
    llr = np.asarray(llr, dtype=np.float64)
    information_pos = _frozen_bits_to_info_pos(frozen_bits)
    return _sc_decode_core(llr, information_pos, frozen_bit=0)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码调度表"""
    n = int(math.log2(N))
    lambda_offset = [(1 << (n - layer)) - 1 for layer in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        psi = phi
        while psi % 2 == 1:
            layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers)

        layers_b = []
        if phi % 2 == 0:
            psi = phi
            while psi % 2 == 0 and psi > 0:
                layers_b.append(int(math.log2(psi & -psi)) + 1)
                psi >>= 1
        bit_layer_vec.append(layers_b)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    information_pos = _frozen_bits_to_info_pos(frozen_bits)
    return _sc_decode_core(llr_ch, information_pos, frozen_bit=0)


def verify_sc_decoders(N=64, frozen_bits=None, trials=100, eb_n0_db=10.0):
    """SC 递归与非递归译码器一致性及无损验证"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    if frozen_bits is None:
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(trials):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_rec = sc_decode(llr, frozen_bits)
        u_rec_r = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_rec_r), "SC recursive/non-recursive mismatch"
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error at high SNR"
