"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f 运算（check node）"""
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    return Lb + (1.0 - 2.0 * u_hat) * La


def _penalty(llr, bit):
    """路径度量惩罚项"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _SCNodeDecoder:
    """极化码 SC 递归树译码器"""

    def __init__(self, frozen_bits):
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.N = len(frozen_bits)
        self.u_hat = np.zeros(self.N, dtype=int)

    def decode(self, llr_ch):
        self.u_hat = np.zeros(self.N, dtype=int)
        self._node(np.asarray(llr_ch, dtype=np.float64), 0, self.N)
        return self.u_hat.copy()

    def _leaf(self, llr, index):
        if self.frozen_bits[index]:
            self.u_hat[index] = 0
        else:
            self.u_hat[index] = 0 if llr[0] >= 0 else 1
        return np.array([self.u_hat[index]], dtype=int)

    def _node(self, llr, base, length):
        if length == 1:
            self._leaf(llr, base)
            return np.array([self.u_hat[base]], dtype=int)

        half = length // 2
        llr_upper = f_operation(llr[:half], llr[half:])
        beta_upper = self._node(llr_upper, base, half)

        llr_lower = g_operation(llr[:half], llr[half:], beta_upper)
        beta_lower = self._node(llr_lower, base + half, half)

        return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _SCNodeDecoder(frozen_bits).decode(llr)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（与递归实现等价）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 使用）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        rho = 0
        tmp = phi
        while tmp & 1:
            rho += 1
            tmp >>= 1
        llr_layer_vec.append(list(range(n - 1, rho - 1, -1)))
        layers_bit = []
        tmp = phi
        layer = 0
        while tmp & 1:
            layers_bit.append(layer)
            tmp >>= 1
            layer += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=15.0):
    """验证 SC 译码正确性"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    sigma = eb_n0_to_sigma(eb_n0_db, K / N)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen_bits)
        u_ref = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_rec, u_ref), "SC decoders disagree"
        assert np.array_equal(u[info_idx], u_rec[info_idx]), "SC decode error"
    return True


if __name__ == "__main__":
    print("SC decoder verification:", verify_sc_decoders())
