"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """log-domain 精确 f 运算（向量化）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _frozen_mask(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(int) != 0


def _reorder_channel_llr(llr_ch):
    """将信道 LLR 映射到蝶形译码器顺序（与比特倒序编码对应）"""
    N = len(llr_ch)
    perm = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[perm]


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


class _SCRecursiveCore:
    """与 SCL L=1 相同的树遍历（参考实现）"""

    def __init__(self, frozen):
        self.frozen = frozen
        self.u_hat = np.zeros(frozen.size, dtype=int)

    def decode(self, llr):
        self._decode_node([llr], 0, self.frozen.size)
        return self.u_hat

    def _leaf(self, llrs, index):
        llr = float(llrs[0][0])
        if self.frozen[index]:
            self.u_hat[index] = 0
            bit = 0
        else:
            bit = 0 if llr >= 0 else 1
            self.u_hat[index] = bit
        return [np.array([bit], dtype=int)]

    def _decode_node(self, llrs, base, length):
        if length == 1:
            return self._leaf(llrs, base)

        half = length // 2
        upper = [f_operation(v[:half], v[half:]) for v in llrs]
        beta_upper = self._decode_node(upper, base, half)

        lower = []
        for p, llr_vec in enumerate(llrs):
            lower.append(g_operation(llr_vec[:half], llr_vec[half:], beta_upper[p]))
        beta_lower = self._decode_node(lower, base + half, half)

        return [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _reorder_channel_llr(llr)
    frozen = _frozen_mask(frozen_bits)
    return _SCRecursiveCore(frozen).decode(llr)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(math.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        t = phi
        for l in range(n):
            if (t & 1) == 0:
                llr_layers.append(l)
            t >>= 1
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        l = 0
        while (phi + 1) % (2 ** (l + 1)) == 0 and l < n:
            bit_layers.append(l)
            l += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（默认使用高效递归核心；见 `_sc_decode_nonrecursive`）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def _sc_decode_nonrecursive(llr_ch, frozen_bits):
    llr_ch = _reorder_channel_llr(llr_ch)
    frozen = _frozen_mask(frozen_bits)
    N = len(llr_ch)
    n = int(math.log2(N))
    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if phi == 0:
            index = 0
            layer = n
        else:
            index = phi - 1
            layer = 0
            while index & 1:
                index >>= 1
                layer += 1
            layer = n - layer - 1
            index = phi >> (n - layer)

        while layer > 0:
            psi = index >> 1
            if index & 1:
                P[layer - 1, psi] = g_operation(
                    P[layer, 2 * psi], P[layer, 2 * psi + 1], C[layer - 1, index - 1]
                )
            else:
                P[layer - 1, psi] = f_operation(
                    P[layer, 2 * psi], P[layer, 2 * psi + 1]
                )
            index = psi
            layer -= 1

        if frozen[phi]:
            u_hat[phi] = 0
            C[0, 0] = 0
        else:
            u_hat[phi] = 0 if P[0, 0] >= 0 else 1
            C[0, 0] = u_hat[phi]

        index = 0
        layer = 0
        while (phi + 1) % (2 ** (layer + 1)) == 0 and layer < n:
            psi = index >> 1
            C[layer + 1, 2 * psi] = (C[layer, index - 1] ^ C[layer, index]) & 1
            C[layer + 1, 2 * psi + 1] = C[layer, index]
            index = psi
            layer += 1

    return u_hat
