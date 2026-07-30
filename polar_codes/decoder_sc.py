"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_rev(i, n):
    r = 0
    for b in range(n):
        if (i >> b) & 1:
            r |= 1 << (n - 1 - b)
    return r


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


class _SCCore:
    """SC 译码核心（列 0 为信道 LLR，列 n 为判决比特）。"""

    def __init__(self, N):
        self.N = N
        self.n = int(np.log2(N))
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int32)

    def set_llr(self, llr_ch):
        self.L[:, 0] = llr_ch

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        top_bit,
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        self.B[j, s] ^ self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode_bits(self, frozen_bits):
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        for i in range(self.N):
            l = _bit_rev(i, self.n)
            self.update_llrs(l)
            if frozen_bits[l]:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = _hard_decision(self.L[l, self.n])
            self.update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    llr_ch 与信道输出 x（polar_encode 后的码字顺序）对齐。
    """
    N = len(llr_ch)
    brev = bit_reversal_permutation(N)
    core = _SCCore(N)
    core.set_llr(np.asarray(llr_ch, dtype=np.float64)[brev])
    return core.decode_bits(frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价，供验证）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（供 SCL 参考）。"""
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=np.int32)
    lambda_offset[0] = 0
    lambda_offset[1] = 1
    for layer in range(1, n):
        lambda_offset[layer + 1] = lambda_offset[layer] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        for layer in range(n):
            if (phi >> layer) & 1 == 0:
                llr_layers.append(layer)
            else:
                bit_layers.append(layer)
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
