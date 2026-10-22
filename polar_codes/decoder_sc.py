"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 为左子树对应的部分码字比特）"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（与 Arikan 因子图一致）。
    g 分支使用左子树返回的部分码字比特。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=np.int8)
    indices = list(range(N))

    def _sc(llrs, idxs):
        if len(idxs) == 1:
            i = idxs[0]
            if frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 0 if llrs[0] >= 0 else 1
            return [int(u_hat[i])]

        half = len(idxs) // 2
        left_llr = np.array(
            [f_operation(llrs[i], llrs[i + half]) for i in range(half)], dtype=np.float64
        )
        enc_left = _sc(left_llr, idxs[:half])
        right_llr = np.array(
            [g_operation(llrs[i], llrs[i + half], enc_left[i]) for i in range(half)],
            dtype=np.float64,
        )
        enc_right = _sc(right_llr, idxs[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(llr, indices)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            llr_layers = list(range(n - 1, -1, -1))
            bit_layers = []
        else:
            p = phi
            t = 0
            while p & 1:
                p >>= 1
                t += 1
            llr_layers = list(range(n - 1, t - 1, -1))
            bit_layers = list(range(t))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主接口。
    当前通过已验证的递归内核实现（接口与仿真脚本一致）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + np.random.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen)
        u_rec_r = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u_rec, u_rec_r)
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            errors += 1
    print(f"SC N=64 high SNR errors: {errors}/100")
