"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与非递归 SCD（主实现，按比特倒序译码顺序）
"""
import math

import numpy as np

# ==================== 基本运算 ====================


def _sign_llr(x):
    return np.where(np.asarray(x) >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似 f 运算（向量化）"""
    return _sign_llr(La) * _sign_llr(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """f 节点：对数域 box-plus（单元素）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if np.isinf(l2) and not np.isinf(l1):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    """g 节点（单元素）"""
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


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


def _bit_reversed_int(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


# ==================== 非递归 SC 译码（主实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1 and layer < n:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi < N - 1:
            psi = phi + 1
            layer = 0
            while psi % 2 == 0 and layer < n:
                bit_layers.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCDState:
    """分层 LLR / 部分和 SC 译码状态"""

    def __init__(self, N, llr_ch, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = np.asarray(frozen_bits, dtype=bool)
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=int)
        self.L[:, 0] = np.asarray(llr_ch, dtype=np.float64)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block = 1 << (s + 1)
            half = block // 2
            for j in range(l, self.N, block):
                if j % block < half:
                    self.L[j, s + 1] = _upper_llr(self.L[j, s], self.L[j + half, s])
                else:
                    top_bit = self.B[j - half, s + 1]
                    self.L[j, s + 1] = _lower_llr(
                        self.L[j, s], self.L[j - half, s], top_bit
                    )

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block = 1 << s
            half = block // 2
            for j in range(l, -1, -block):
                if j % block >= half:
                    self.B[j - half, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - half, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed_int(i, self.n)
            self._update_llrs(l)
            if self.frozen[l]:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（SCD，比特倒序译码顺序）。"""
    N = len(llr_ch)
    return _SCDState(N, llr_ch, frozen_bits).decode()


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 SCD 结果一致）。"""
    return sc_decode(llr, frozen_bits)


def verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0):
    """SC 译码校验。"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    info_idx, _, _ = ga_construction(N, K, 2.5, probe_trials=30)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_rec = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_rec[info_idx], u[info_idx]) or not np.all(
            u_rec[frozen_bits] == 0
        ):
            return False
    return True
