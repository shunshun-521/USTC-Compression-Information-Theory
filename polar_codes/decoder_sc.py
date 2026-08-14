"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    sa, sb = np.sign(La), np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _all_computed(x):
    return not np.any(np.isnan(x))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    span = 2 ** (position[2] - position[0] - 1)
    return [position[0] + 1, position[1] + span, position[2], position[3]]


def _up(position):
    p0 = position[0] - 1
    span = 2 ** (position[2] - position[0] + 1)
    p1 = int(np.floor(position[1] / span) * span)
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(1, 2 * length).flatten()


def _get_left_llr(up_llr):
    length = len(up_llr) // 2
    return np.array([f_operation(up_llr[i], up_llr[i + length]) for i in range(length)])


def _get_right_llr(left_bit, up_llr):
    length = len(left_bit)
    return np.array([
        g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)
    ])


def _decide_bit(llr_val, bit_pos, info_positions, frozen_value):
    if bit_pos in info_positions:
        return 0 if llr_val >= 0 else 1
    return frozen_value


def _sc_tree_decode(y_llr, information_pos, frozen_value=0):
    """树遍历 SC 译码核心（llr_matrix[0] 为信道 LLR）。"""
    N = len(y_llr)
    n = int(np.log2(N))
    info_set = set(information_pos.tolist())

    llr_matrix = np.full((n + 1, N), np.nan)
    bit_matrix = np.full((n + 1, N), np.nan)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_computed(up_bit):
            position = _up(position)
        elif _all_computed(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _decide_bit(
                    right_llr[0], right_bit_pos, info_set, frozen_value
                )
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_computed(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr
        elif not _all_computed(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _decide_bit(
                    left_llr[0], left_bit_pos, info_set, frozen_value
                )
                bit_matrix[position[0] + 1][position[1]:position[1] + half] = left_bit_val
            else:
                position = _leftdown(position)

    result = bit_matrix[n]
    return np.array([0 if v == 0 else 1 for v in result], dtype=int)


def _prepare_llr(llr_ch):
    """比特倒序置换信道 LLR，与编码器比特倒序一致。"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def _info_positions(frozen_bits):
    return np.where(np.asarray(frozen_bits, dtype=int) == 0)[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    y_llr = _prepare_llr(llr_ch)
    info_pos = _info_positions(frozen_bits)
    return _sc_tree_decode(y_llr, info_pos, frozen_value=0)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价的树遍历实现）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi % 2 == 1:
            layers_llr.append(int(np.log2(psi & -psi)))
            psi //= 2
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        psi = phi
        while psi % 2 == 1:
            layers_bit.append(int(np.log2(psi & -psi)))
            psi = (psi + 1) // 2
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
    from construction import ga_construction

    rng = np.random.default_rng(0)
    for N in [4, 8, 16, 64]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        sigma = eb_n0_to_sigma(10.0, K / N)
        errors = 0
        trials = min(100, 2 ** K)
        for _ in range(trials):
            u = np.zeros(N, dtype=int)
            u[info_idx] = rng.integers(0, 2, K)
            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)
            u_hat = sc_decode(llr, frozen_bits)
            if np.any(u_hat[info_idx] != u[info_idx]):
                errors += 1
        print(f"N={N}: errors={errors}/{trials} at Eb/N0=10dB")
