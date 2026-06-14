"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：right - (2*u - 1) * left"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0 or u_hat.size == 1:
        u = int(u_hat.flat[0]) if u_hat.size else int(u_hat)
        return Lb - (2 * u - 1) * La
    return Lb - (2 * u_hat - 1) * La


def _compute_left_alpha(llr):
    half = len(llr) // 2
    left = llr[:half]
    right = llr[half:]
    return f_operation(left, right)


def _compute_right_alpha(llr, left_bits):
    half = len(llr) // 2
    left = llr[:half]
    right = llr[half:]
    return right - (2 * left_bits - 1) * left


def _compute_encoding_step(level, n, source):
    result = source.copy()
    step = 1 << (n - level - 1)
    groups = 1 << level
    for g in range(groups):
        start = 2 * g * step
        for p in range(step):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]
    return result


def _position_state(position, n):
    bits = np.unpackbits(np.array([position], dtype=np.uint32).byteswap().view(np.uint8))
    return bits[-n:]


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
            u_hat[idx] = 0 if frozen_bits[idx] or llr_node[0] >= 0 else 1
            return
        half = n // 2
        llr_left = _compute_left_alpha(llr_node)
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = _compute_right_alpha(llr_node, u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助信息（供文档/扩展使用）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        state = _position_state(phi, n)
        prev = np.ones(n, dtype=int)
        llr_layers = [i for i in range(n) if state[i] != prev[i]]
        bit_layers = list(range(n))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return list(range(N)), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 按码字自然顺序输入。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    intermediate_llr = [llr_ch.copy()]
    length = N // 2
    while length > 0:
        intermediate_llr.append(np.zeros(length, dtype=np.float64))
        length //= 2

    intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
    current_state = np.zeros(n, dtype=np.int8)
    previous_state = np.ones(n, dtype=np.int8)

    for position in range(N):
        current_state = _position_state(position, n)

        for i in range(1, n + 1):
            llr = intermediate_llr[i - 1]
            if current_state[i - 1] == previous_state[i - 1]:
                continue
            if current_state[i - 1] == 0:
                intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (1 << (n - i))
                left_bits = intermediate_bits[i][start:end]
                intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

        if frozen_bits[position]:
            decision = 0
        else:
            decision = 1 if intermediate_llr[-1][0] < 0 else 0

        intermediate_bits[-1][position] = decision
        for i in range(n - 1, -1, -1):
            intermediate_bits[i] = _compute_encoding_step(
                i, n, intermediate_bits[i + 1]
            )

        previous_state = current_state.copy()

    return intermediate_bits[-1].astype(int)


def verify_sc_decoders(N=64, K=32, num_frames=100):
    """SC 译码验证"""
    from channel import bpsk_modulate, compute_llr
    from construction import ga_construction
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-8)
        u_rec = sc_decode(llr, frozen_bits)
        assert np.array_equal(u_rec[info_idx], u[info_idx]), "SC decode error"


if __name__ == "__main__":
    verify_sc_decoders()
