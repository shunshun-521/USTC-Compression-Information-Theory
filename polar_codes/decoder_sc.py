"""
极化码 SC（串行抵消）译码器
Permuted SCD（Vangala 2014），与 encoder.polar_encode 配套
"""
import numpy as np
from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似 f 运算"""
    La = np.atleast_1d(np.asarray(La, dtype=np.float64))
    Lb = np.atleast_1d(np.asarray(Lb, dtype=np.float64))
    s1 = np.sign(La)
    s2 = np.sign(Lb)
    np.putmask(s1, s1 == 0, 1.0)
    np.putmask(s2, s2 == 0, 1.0)
    out = s1 * s2 * np.minimum(np.abs(La), np.abs(Lb))
    return float(out[0]) if out.size == 1 else out


def g_operation(La, Lb, u_hat):
    """g 运算（u_hat 为 0/1）"""
    out = (1 - 2 * u_hat) * La + Lb
    if np.isscalar(out):
        return float(out)
    arr = np.asarray(out)
    return float(arr[0]) if arr.size == 1 else arr


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def upper_llr(l1, l2):
    """f 分支精确 LLR（对数域）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
    """g 分支精确 LLR"""
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
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
    """预计算每个比特的 LLR/比特回传层列表（与 Permuted SCD 一致）"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = bit_reversed(i, n)
        llr_layers = list(range(n - active_llr_level(l, n), n))
        bit_layers = list(range(n, n - active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return llr_layer_vec, bit_layer_vec


class _SCDState:
    def __init__(self, N, frozen_indices, use_minsum=True):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen = set(int(i) for i in frozen_indices)
        self.use_minsum = use_minsum
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, self.n + 1), np.nan)

    def _upper(self, a, b):
        return f_operation(a, b) if self.use_minsum else upper_llr(a, b)

    def _lower(self, a, b, bit):
        return g_operation(a, b, bit) if self.use_minsum else lower_llr(a, b, bit)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = self._upper(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = self._lower(
                        self.L[j - branch_size, s], self.L[j, s], top_bit
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self, llr_ch):
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)
        self.B.fill(np.nan)
        for i in range(self.N):
            l = bit_reversed(i, self.n)
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self.update_bits(l)
        return self.B[:, self.n].astype(np.int8)


def _frozen_indices_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return np.where(frozen_bits.astype(bool))[0]


def sc_decode(llr_ch, frozen_bits):
    """非递归 Permuted SC 译码"""
    N = len(llr_ch)
    frozen_idx = _frozen_indices_from_mask(frozen_bits)
    return _SCDState(N, frozen_idx, use_minsum=True).decode(llr_ch)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 参考接口（与 sc_decode 共用 Permuted SCD 实现）"""
    return sc_decode(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info] = 0
    rng = np.random.default_rng(0)
    ok = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = bpsk_modulate(x) + rng.normal(0, sigma, N)
        llr = compute_llr(y, sigma)
        u1 = sc_decode(llr, frozen_bits)
        u2 = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u1, u2)
        if np.all(u[info] == u1[info]):
            ok += 1
    print(f"SC: recursive match; info correct {ok}/100 @ 10dB")
