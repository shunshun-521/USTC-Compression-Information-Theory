"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效置换 SC 实现）
"""
import numpy as np

LLR_MAX = 20.0


def clip_llr(x):
    return np.clip(x, -LLR_MAX, LLR_MAX)


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(btm_llr, top_llr, b):
    """下分支 LLR 更新（参数顺序与置换 SC 一致：先 bottom 后 top）"""
    if b == 0:
        if np.isinf(btm_llr) or np.isinf(top_llr):
            return np.inf
        return btm_llr + top_llr
    return btm_llr - top_llr


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


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（向量化接口，供 BP 等模块使用）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return clip_llr(np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb)))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return clip_llr((1 - 2 * u_hat) * La + Lb)


class _SCD:
    """置换 SC 译码器内核（基于 permuted successive cancellation）"""

    def __init__(self, N, n, llr_ch, frozen_set):
        self.N = N
        self.n = n
        self.frozen_set = frozen_set
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = clip_llr(llr_ch)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s],
                        self.L[j - branch_size, s],
                        int(self.B[j - branch_size, s + 1]),
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            if l in self.frozen_set:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（置换 SC）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])
    return _SCD(N, n, llr_ch, frozen_set).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用置换 SC 实现）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（兼容 SCL 模块接口）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bit_reversed(phi, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        if l >= N / 2:
            bit_start = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(bit_start, n))[::-1])
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec
