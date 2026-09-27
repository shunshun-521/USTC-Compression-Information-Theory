"""
极化码 SC（串行抵消）译码器
非递归实现（参考 Permuted SCD），含递归接口别名
"""
import math
import numpy as np

from encoder import polar_encode


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reverse(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


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


def _upper_llr(l1, l2):
    return f_operation(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


class _SCDState:
    def __init__(self, llr_ch, frozen_bits):
        self.llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.N = len(llr_ch)
        self.n = int(math.log2(self.N))
        self.L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = self.llr_ch

    def update_llrs(self, l):
        start = self.n - _active_llr_level(l, self.n)
        for s in range(start, self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = _upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    self.L[j, s + 1] = _lower_llr(btm_llr, top_llr, int(top_bit))

    def update_bits(self, l):
        if l < self.N / 2:
            return
        end = self.n - _active_bit_level(l, self.n)
        for s in range(self.n, end, -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    b_j = self.B[j, s]
                    b_jt = self.B[j - branch_size, s]
                    if np.isnan(b_j):
                        b_j = 0
                    if np.isnan(b_jt):
                        b_jt = 0
                    self.B[j - branch_size, s - 1] = int(b_j) ^ int(b_jt)
                    self.B[j, s - 1] = int(b_j)

    def run(self):
        for i in range(self.N):
            l = _bit_reverse(i, self.n)
            self.update_llrs(l)
            if self.frozen_bits[l]:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(np.int32)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码。"""
    return _SCDState(llr_ch, frozen_bits).run()


def sc_decode_recursive(llr, frozen_bits):
    """递归接口（与主实现等价）。"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """SCL 使用的分层索引（保留接口）。"""
    n = int(math.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        start_layer = None
        for layer in range(n):
            if ((phi >> layer) & 1) == 0:
                start_layer = layer
                break
        if start_layer is None:
            layers_llr = list(range(n - 1, -1, -1))
        else:
            layers_llr = list(range(n - 1, start_layer - 1, -1))
        llr_layer_vec.append(layers_llr)
        layers_bit = [layer for layer in range(n) if (phi >> layer) & 1]
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _llr_update_layer(P, C, layer, n, lambda_offset):
    sp = lambda_offset[layer]
    pm = sp
    block = 2 * sp
    for block_start in range(0, 2 ** n, block):
        for j in range(pm):
            idx_u = block_start + j
            idx_v = block_start + j + sp
            La = P[layer + 1, idx_u]
            Lb = P[layer + 1, idx_v]
            parity = (idx_v // sp) % 2
            if parity == 1:
                u_partial = C[layer, idx_u]
                P[layer, idx_u] = g_operation(La, Lb, u_partial)
            else:
                P[layer, idx_u] = f_operation(La, Lb)


def _bit_update_layer(C, layer, n, lambda_offset):
    sp = lambda_offset[layer]
    pm = sp
    block = 2 * sp
    for block_start in range(0, 2 ** n, block):
        for j in range(pm):
            idx_u = block_start + j
            idx_v = block_start + j + sp
            C[layer + 1, idx_u] = C[layer, idx_u]
            C[layer + 1, idx_v] = (C[layer, idx_u] + C[layer, idx_v]) % 2
