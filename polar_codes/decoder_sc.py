"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（硬件友好型，sign(0)=+1）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_computed(x):
    return not np.any(np.isnan(x))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    span = 2 ** (position[2] - 1 - position[0])
    return [position[0] + 1, position[1] + span, position[2], position[3]]


def _up(position):
    span = 2 ** (position[2] - position[0] + 1)
    return [
        position[0] - 1,
        int(math.floor(position[1] / span) * span),
        position[2],
        position[3],
    ]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.zeros((1, 2 * length))
    temp[0, :length] = (left_bit + right_bit) % 2
    temp[0, length:] = right_bit
    return temp[0]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)

    def decode_block(llr_blk, pos, length):
        if length == 1:
            if frozen_bits[pos]:
                return [0]
            return [0 if llr_blk[0] >= 0 else 1]
        half = length // 2
        llr_l = f_operation(llr_blk[:half], llr_blk[half:])
        u_l = decode_block(llr_l, pos, half)
        llr_r = g_operation(llr_blk[:half], llr_blk[half:], np.array(u_l, dtype=int))
        u_r = decode_block(llr_r, pos + half, half)
        return u_l + u_r

    return np.array(decode_block(llr, 0, N), dtype=int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（因子图树遍历）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    info_indices = np.where(~frozen_bits)[0]
    frozen_val = 0

    N = len(llr_ch)
    n = int(math.log2(N))

    llr_matrix = np.ones((n + 1, N), dtype=np.float64)
    llr_matrix[:] = np.nan
    bit_matrix = llr_matrix.copy()
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        layer, col, max_layer, _ = position
        span = 2 ** (max_layer - layer)
        up_llr = llr_matrix[layer][col:col + span]
        up_bit = bit_matrix[layer][col:col + span]
        half = span // 2
        left_llr = llr_matrix[layer + 1][col:col + half]
        left_bit = bit_matrix[layer + 1][col:col + half]
        right_llr = llr_matrix[layer + 1][col + half:col + span]
        right_bit = bit_matrix[layer + 1][col + half:col + span]

        if _all_computed(up_bit):
            position = _up(position)
        else:
            if _all_computed(right_bit):
                up_bit_new = _get_up_bit(left_bit, right_bit)
                bit_matrix[layer][col:col + span] = up_bit_new
            else:
                if _all_computed(right_llr):
                    if layer == max_layer - 1:
                        right_pos = col + 1
                        if frozen_bits[right_pos]:
                            right_bit_val = frozen_val
                        else:
                            right_bit_val = 0 if right_llr[0] > 0 else 1
                        bit_matrix[layer + 1][col + half] = right_bit_val
                    else:
                        position = _rightdown(position)
                else:
                    if _all_computed(left_bit):
                        right_llr_new = g_operation(up_llr[:half], up_llr[half:], left_bit)
                        llr_matrix[layer + 1][col + half:col + span] = right_llr_new
                    else:
                        if not _all_computed(left_llr):
                            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
                            llr_matrix[layer + 1][col:col + half] = left_llr_new
                        else:
                            if layer == max_layer - 1:
                                left_pos = col
                                if frozen_bits[left_pos]:
                                    left_bit_val = frozen_val
                                else:
                                    left_bit_val = 0 if left_llr[0] >= 0 else 1
                                bit_matrix[layer + 1][col] = left_bit_val
                            else:
                                position = _leftdown(position)

    return bit_matrix[n].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers, p = [], phi
        while p % 2 == 1:
            layers.append(len(layers))
            p >>= 1
        llr_layer_vec.append(layers)
        layers_b, psi = [], phi + 1
        while psi % 2 == 0:
            layers_b.append(len(layers_b))
            psi >>= 1
        bit_layer_vec.append(layers_b)
    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    rng = np.random.default_rng(0)
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    print(f"SC test (Eb/N0=10dB): {errors} errors in 100 frames")

    # 无损验证
    noiseless_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x).astype(float), 1.0) * 1e3
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            noiseless_err += 1
    print(f"SC noiseless test: {noiseless_err} errors in 100 frames (expect 0)")
    assert noiseless_err == 0, "SC decoder failed noiseless test"
