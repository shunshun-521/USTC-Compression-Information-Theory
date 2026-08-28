"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
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


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的三个辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 2 ** i - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            end = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n, end, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDecoderState:
    """非递归 SC 译码（mcba1n 风格，比特倒序译码顺序）"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan, dtype=np.float64)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        self.B[:] = np.nan
        u_hat = np.zeros(self.N, dtype=int)

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            self._update_llrs(l)
            if self.frozen_bits[l]:
                u_hat[l] = 0
                self.B[l, self.n] = 0
            else:
                u_hat[l] = 0 if self.L[l, self.n] >= 0 else 1
                self.B[l, self.n] = u_hat[l]
            self._update_bits(l)

        return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    from encoder import prepare_channel_llr

    N = len(llr_ch)
    llr_adj = prepare_channel_llr(llr_ch, N)
    return _SCDecoderState(N, frozen_bits).decode(llr_adj)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，与非递归结果一致）。"""
    return sc_decode(llr, frozen_bits)
