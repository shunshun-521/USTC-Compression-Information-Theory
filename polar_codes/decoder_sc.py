"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，因子树遍历）
"""
import numpy as np


def _sign_hf(x):
    """硬件友好符号：sign(0) = +1。"""
    s = np.sign(x)
    if np.isscalar(s):
        return 1.0 if s == 0 else float(s)
    return np.where(s == 0, 1.0, s)


def f_operation(La, Lb):
    """min-sum f 运算。"""
    return _sign_hf(La) * _sign_hf(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1 - 2 * u_hat) * La + Lb


def _all_filled(row):
    return not np.any(np.isnan(row))


def _up_position(position):
    position[0] -= 1
    position[1] = int(
        np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
        * (2 ** (position[2] - position[0] + 1))
    )


def _decide_bit(llr, pos, frozen_bits):
    if frozen_bits[pos]:
        return 0
    return 0 if llr >= 0 else 1


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（因子树遍历，与 u@G 编码配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = llr_ch
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        half = span // 2
        p0, p1 = position[0], position[1]

        up_llr = llr_matrix[p0][p1: p1 + span]
        up_bit = bit_matrix[p0][p1: p1 + span]
        left_llr = llr_matrix[p0 + 1][p1: p1 + half]
        left_bit = bit_matrix[p0 + 1][p1: p1 + half]
        right_llr = llr_matrix[p0 + 1][p1 + half: p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + half: p1 + span]

        if _all_filled(up_bit):
            _up_position(position)
        elif _all_filled(right_bit):
            up_val = np.zeros(span)
            up_val[:half] = (left_bit + right_bit) % 2
            up_val[half:] = right_bit
            bit_matrix[p0][p1: p1 + span] = up_val
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                pos = p1 + half
                bit_matrix[p0 + 1][pos] = _decide_bit(right_llr[0], pos, frozen_bits)
            else:
                position[0] += 1
                position[1] += half
        elif _all_filled(left_bit):
            right_new = np.array(
                [g_operation(up_llr[i], up_llr[i + half], left_bit[i]) for i in range(half)]
            )
            llr_matrix[p0 + 1][p1 + half: p1 + span] = right_new
        elif _all_filled(left_llr):
            if position[0] == position[2] - 1:
                pos = p1
                bit_matrix[p0 + 1][pos] = _decide_bit(left_llr[0], pos, frozen_bits)
            else:
                position[0] += 1
        else:
            llr_matrix[p0 + 1][p1: p1 + half] = f_operation(up_llr[:half], up_llr[half:])

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(llr, dtype=np.float64)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = _decide_bit(llr_node[0], idx, frozen_bits)
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算辅助向量（SCL 兼容接口）。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        psi, layer = phi, 0
        while psi % 2 == 1:
            psi >>= 1
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        psi, layer = phi, 0
        while psi > 0 and psi % 2 == 0:
            psi >>= 1
            layer += 1
        bit_layer_vec.append(list(range(layer)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0, seed=42):
    """在极低噪声下验证 SC 译码器。"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    rng = np.random.default_rng(seed)
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)

        u_rec = sc_decode(llr, frozen_bits)
        assert np.array_equal(u, u_rec), "SC 译码错误"

    if N <= 8:
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
        assert np.array_equal(sc_decode(llr, frozen_bits), sc_decode_recursive(llr, frozen_bits))

    return True


if __name__ == "__main__":
    verify_sc_decoders()
    print("SC 译码器校验通过")
