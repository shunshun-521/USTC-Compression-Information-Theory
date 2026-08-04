"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1 = np.where(s1 == 0, 1, s1)
    s2 = np.where(s2 == 0, 1, s2)
    return s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _all_decided(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1, p2, p3]


def _rightdown(position):
    p0, p1, p2, p3 = position
    return [p0 + 1, p1 + 2 ** (p2 - p0 - 1), p2, p3]


def _up(position):
    p0, p1, p2, p3 = position
    p1_t = int(np.floor(p1 / (2 ** (p2 - p0 + 1))) * (2 ** (p2 - p0 + 1)))
    return [p0 - 1, p1_t, p2, p3]


def _get_up_bit(left_bit, right_bit):
    length = len(left_bit)
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    return temp.reshape(2 * length)


def sc_decode_step(llr_matrix, bit_matrix, frozen_bits, split_pos):
    """运行 SC 直到 split_pos 判决完成，返回矩阵与叶节点 LLR"""
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    position = _get_resume_position(bit_matrix)

    leaf_llr = 0.0
    while np.isnan(bit_matrix[n][split_pos]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
            continue
        if _all_decided(right_bit):
            bit_matrix[position[0]][position[1]:position[1] + span] = _get_up_bit(left_bit, right_bit)
            continue
        if _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if frozen_bits[right_bit_pos]:
                    val = 0
                else:
                    val = 0 if right_llr[0] >= 0 else 1
                    if right_bit_pos == split_pos:
                        leaf_llr = right_llr[0]
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = val
            else:
                position = _rightdown(position)
            continue
        if _all_decided(left_bit):
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = g_operation(
                up_llr[:half], up_llr[half:], left_bit
            )
            continue
        if not _all_decided(left_llr):
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = f_operation(
                up_llr[:half], up_llr[half:]
            )
            continue
        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            if frozen_bits[left_bit_pos]:
                val = 0
            else:
                val = 0 if left_llr[0] >= 0 else 1
                if left_bit_pos == split_pos:
                    leaf_llr = left_llr[0]
            bit_matrix[position[0] + 1][position[1]:position[1] + half] = val
        else:
            position = _leftdown(position)

    return llr_matrix, bit_matrix, leaf_llr


def _get_resume_position(bit_matrix):
    N = bit_matrix.shape[1]
    n = int(math.log2(N))
    detect_array = bit_matrix[n]
    detect = -1
    for i in range(N):
        if np.isnan(detect_array[i]):
            detect = i - 1
            break
    if detect == -1:
        return [0, 0, n, N]
    if detect % 2 == 0:
        return [n - 1, detect, n, N]
    return [n - 1, detect - 1, n, N]


def path_metric_update(llr_val, u_bit):
    hard = 0 if llr_val >= 0 else 1
    return 0.0 if u_bit == hard else abs(llr_val)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（矩阵化树遍历）"""
    y_llr = np.asarray(llr_ch, dtype=np.float64)
    N = len(y_llr)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_decided(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1]:position[1] + span]
        up_bit = bit_matrix[position[0]][position[1]:position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1]:position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1]:position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half:position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half:position[1] + span]

        if _all_decided(up_bit):
            position = _up(position)
            continue

        if _all_decided(right_bit):
            up_bit_new = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1]:position[1] + span] = up_bit_new
            continue

        if _all_decided(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                if frozen_bits[right_bit_pos]:
                    right_bit_val = 0
                else:
                    right_bit_val = 0 if right_llr[0] >= 0 else 1
                bit_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
            continue

        if _all_decided(left_bit):
            right_llr_new = g_operation(up_llr[:half], up_llr[half:], left_bit)
            llr_matrix[position[0] + 1][position[1] + half:position[1] + span] = right_llr_new
            continue

        if not _all_decided(left_llr):
            left_llr_new = f_operation(up_llr[:half], up_llr[half:])
            llr_matrix[position[0] + 1][position[1]:position[1] + half] = left_llr_new
            continue

        if position[0] == position[2] - 1:
            left_bit_pos = position[1]
            if frozen_bits[left_bit_pos]:
                left_bit_val = 0
            else:
                left_bit_val = 0 if left_llr[0] >= 0 else 1
            bit_matrix[position[0] + 1][position[1]:position[1] + half] = left_bit_val
        else:
            position = _leftdown(position)

    u_hat = np.nan_to_num(bit_matrix[n], nan=0).astype(int)
    u_hat[frozen_bits] = 0
    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树形分裂实现）"""
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, offset):
        n = len(llr_node)
        if n == 1:
            idx = offset
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, offset)
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_hat[offset:offset + half])
        decode_node(llr_right, offset + half)

    decode_node(np.asarray(llr, dtype=np.float64), 0)
    u_hat[frozen_bits] = 0
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers, p, layer = [], phi, 0
        while (p % 2 == 0) and layer < n:
            layers.append(layer)
            p >>= 1
            layer += 1
        llr_layer_vec.append(layers)
        bit_layers, p, layer = [], phi, 0
        while (p % 2 == 1) and layer < n:
            bit_layers.append(layer)
            p >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return sc_decode_nonrecursive(llr_ch, frozen_bits)


def channel_llr_for_decode(llr_ch):
    """将信道 LLR 转换为 SC 译码器输入顺序"""
    from encoder import bit_reversal_permutation
    llr = np.asarray(llr_ch, dtype=np.float64)
    return llr[bit_reversal_permutation(len(llr))]


def verify_sc_decoder(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码验证"""
    from construction import ga_construction
    from encoder import polar_encode_natural
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode_natural(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_hat[info_idx], u[info_idx]), "SC decode error"

    print(f"SC verification passed: N={N}, K={K}, {num_frames} frames at {eb_n0_db}dB")


if __name__ == "__main__":
    verify_sc_decoder()
