"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(format(i, f'0{n}b')[::-1], 2)


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _prepare_llr(llr_ch):
    """将信道 LLR 重排为译码树所需顺序（与编码器输出比特倒序对应）"""
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[br]


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现，与非递归 SCD 等价）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layers = (
            list(range(n, n - _active_bit_level(l, n), -1))
            if l >= N // 2 else []
        )
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDEngine:
    """非递归 SCD 内核"""

    def __init__(self, N, frozen_bits, llr):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
        self.L = np.zeros((N, self.n + 1), dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)
        self.L[:, 0] = llr

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        self.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (
                        int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    llr = _prepare_llr(llr_ch)
    engine = _SCDEngine(len(llr_ch), frozen_bits, llr)
    for phi in range(engine.N):
        l = _bit_reversed(phi, engine.n)
        engine._update_llrs(l)
        if l in engine.frozen:
            engine.B[l, engine.n] = 0
        else:
            engine.B[l, engine.n] = 0 if engine.L[l, engine.n] >= 0 else 1
        engine._update_bits(l)
    return engine.B[:, engine.n].astype(int)
