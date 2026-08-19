"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _compute_left_alpha(llr):
    """f 运算：llr 前半与后半逐对合并"""
    half = len(llr) // 2
    return f_operation(llr[:half], llr[half:])


def _compute_right_alpha(llr, left_bits):
    """g 运算（向量化）"""
    half = len(llr) // 2
    left = llr[:half]
    right = llr[half:]
    return right - (2.0 * left_bits - 1.0) * left


def _compute_encoding_step(level, n, source, result):
    """单步极化编码（用于比特回传）"""
    step = 2 ** (n - level - 1)
    groups = 2 ** level
    out = result.copy()
    for g in range(groups):
        start = 2 * g * step
        for p in range(step):
            out[p + start] = source[p + start] ^ source[p + start + step]
            out[p + start + step] = source[p + start + step]
    return out


def _position_state(position, n):
    """将位置索引展开为 n 位二进制状态（MSB 在索引 0）"""
    bits = np.zeros(n, dtype=np.int8)
    for i in range(n):
        bits[i] = (position >> (n - 1 - i)) & 1
    return bits


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（自然序逐位译码，参考 python-polar-coding）。
    frozen_bits: 1=冻结位，0=信息位
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    mask = 1 - frozen_bits  # 1=信息位，0=冻结位

    intermediate_llr = [llr_ch.copy()]
    length = N // 2
    while length > 0:
        intermediate_llr.append(np.zeros(length, dtype=np.float64))
        length //= 2

    intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
    previous_state = np.ones(n, dtype=np.int8)

    for position in range(N):
        current_state = _position_state(position, n)

        for i in range(1, n + 1):
            if current_state[i - 1] == previous_state[i - 1]:
                continue
            llr = intermediate_llr[i - 1]
            if current_state[i - 1] == 0:
                intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (2 ** (n - i))
                left_bits = intermediate_bits[i][start:end]
                intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

        if mask[position] == 1:
            intermediate_bits[n][position] = (
                1 if intermediate_llr[n][0] < 0 else 0
            )
        else:
            intermediate_bits[n][position] = 0

        for i in range(n - 1, -1, -1):
            intermediate_bits[i] = _compute_encoding_step(
                i, n, intermediate_bits[i + 1], intermediate_bits[i]
            )

        previous_state = current_state.copy()

    return intermediate_bits[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
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
    """预计算非递归 SC 辅助向量（供 SCL 使用）"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while p % 2 == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)
        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            bit_layers = []
            p = phi
            while p % 2 == 1:
                bit_layers.append(int(math.log2(p & -p)))
                p >>= 1
            bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors_n, errors_r, mismatch = 0, 0, 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        u_hat_r = sc_decode_recursive(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_hat, u_hat_r):
            mismatch += 1
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors_n += 1
    print(f"SC test N=64 errors={errors_n}/100, mismatch={mismatch}")
