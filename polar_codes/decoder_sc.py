"""
极化码 SC（串行抵消）译码器
"""
import numpy as np


def f_operation(La, Lb):
    """f 运算（对数域精确形式）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 可为标量或与 La 同形的部分和数组）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _SCRecursiveCore:
    """极化树递归 SC 内核（单路径，与 Kronecker 编码一致）"""

    def __init__(self, frozen_bits):
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.N = self.frozen.size
        self.u_hat = np.zeros(self.N, dtype=np.int32)

    def decode(self, llr_ch):
        llr = np.asarray(llr_ch, dtype=np.float64)
        self.u_hat[:] = 0
        self._node([llr], 0, self.N)
        return self.u_hat.copy()

    def _leaf(self, llrs, index):
        llr = float(llrs[0][0])
        if self.frozen[index]:
            bit = 0
        else:
            bit = 0 if llr >= 0 else 1
        self.u_hat[index] = bit
        return np.array([bit], dtype=np.int32)

    def _node(self, llrs, base, length):
        if length == 1:
            beta = self._leaf(llrs, base)
            return beta

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
        beta_upper = self._node(upper, base, half)

        a = llrs[0][:half]
        b = llrs[0][half:]
        lower = [g_operation(a, b, beta_upper)]
        beta_lower = self._node(lower, base + half, half)

        return np.concatenate([beta_upper ^ beta_lower, beta_lower])


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _SCRecursiveCore(frozen_bits).decode(llr)


def precompute_sc_indices(N):
    """预计算非递归 SC 所需索引（供扩展实现使用）"""
    import math

    n = int(math.log2(N))
    lambda_offset = [1 << (n - i) for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        ll_layers = []
        idx = phi
        for layer in range(n - 1, -1, -1):
            if (idx & 1) == 0:
                ll_layers.append(layer)
            idx >>= 1
        llr_layer_vec.append(ll_layers)

        bit_layers = []
        idx = phi
        for layer in range(n):
            if (idx & 1) == 1:
                bit_layers.append(layer)
            idx >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（递归单路径实现，O(N log N)）"""
    frozen = np.asarray(frozen_bits, dtype=bool)
    if frozen.dtype != bool:
        frozen = frozen.astype(bool) if frozen.dtype != np.int8 else (frozen != 0)
    return sc_decode_recursive(llr_ch, frozen)
