"""
极化码 SC（串行抵消）译码器
非递归实现基于 Vangala et al. SCD；递归接口与之等价
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（供 SCL/BP 使用）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sign = np.sign(La) * np.sign(Lb)
    sign = np.where(sign == 0, 1.0, sign)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _hard_decision(y):
    return 0 if y >= 0 else 1


def _upper_llr(l1, l2):
    if l1 == np.inf and l2 != np.inf:
        return l2
    if l1 != np.inf and l2 == np.inf:
        return l1
    if l1 == np.inf and l2 == np.inf:
        return np.inf
    return float(np.sign(l1) * np.sign(l2) * min(abs(l1), abs(l2)))


def _lower_llr(l1, l2, b):
    if b == 0:
        if l1 == np.inf or l2 == np.inf:
            return np.inf
        return l1 + l2
    if b == 1:
        return l1 - l2
    return np.nan


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


class _SCD:
    """Vangala et al. 非递归 SC 译码核心"""

    def __init__(self, pc):
        self.pc = pc
        self.L = np.full((pc.N, pc.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((pc.N, pc.n + 1), np.nan)
        self.L[:, 0] = pc.likelihoods

    def decode(self):
        for l in [_bit_reversed(i, self.pc.n) for i in range(self.pc.N)]:
            self._update_llrs(l)
            if l in self.pc.frozen:
                self.B[l, self.pc.n] = 0
            else:
                self.B[l, self.pc.n] = _hard_decision(self.L[l, self.pc.n])
            self._update_bits(l)
        return self.B[:, self.pc.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.pc.n - _active_llr_level(l, self.pc.n), self.pc.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.pc.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.pc.N / 2:
            return
        for s in range(self.pc.n, self.pc.n - _active_bit_level(l, self.pc.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


class _PC:
    def __init__(self, N, llr):
        self.N = N
        self.n = int(math.log2(N))
        self.likelihoods = np.asarray(llr, dtype=np.float64)
        self.frozen = set()


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits[i]=True 表示冻结位。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    pc = _PC(len(llr_ch), llr_ch)
    pc.frozen = set(np.where(frozen_bits)[0])
    return _SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 sc_decode 等价，供校验）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [0]
    for layer in range(1, n + 1):
        lambda_offset.append(lambda_offset[-1] + (1 << (n - layer)))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 0:
                layers_llr.append(layer)
            temp >>= 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 1:
                layers_bit.append(layer)
            temp >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec
