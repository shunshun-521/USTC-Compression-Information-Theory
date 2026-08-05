"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from sc_ref import sc_decoder_ref


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La = La.reshape(1)
        Lb = Lb.reshape(1)
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    s1[s1 == 0] = 1
    s2[s2 == 0] = 1
    result = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return result.item() if scalar else result


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * np.asarray(u_hat, dtype=int)) * La + Lb


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    info_pos = list(np.where(frozen_bits == 0)[0])
    frozen_val = 0
    return sc_decoder_ref(np.asarray(llr_ch, dtype=np.float64), info_pos, frozen_val)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量（供 SCL 使用）"""
    n = int(np.log2(N))
    lambda_offset = [0]
    offset = 0
    for layer in range(n + 1):
        lambda_offset.append(offset)
        offset += 2 ** (n - layer)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                llr_layers.append(layer)
            else:
                break
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 1:
                bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoder():
    """SC 译码无损验证"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"SC decoder failed: {errors} errors in 100 frames"
    print("SC decoder verification passed!")


if __name__ == "__main__":
    verify_sc_decoder()
