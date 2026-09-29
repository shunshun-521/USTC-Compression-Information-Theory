"""
极化码 SC（串行抵消）译码器
"""
import math
import numpy as np


def f_operation(La, Lb):
    """min-sum f（SCL/BP 使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _llr_check_node_operation(llr_1, llr_2):
    """f 运算（boxplus，标量）"""
    if abs(llr_1) > 30 and abs(llr_2) > 30:
        return np.sign(llr_1) * np.sign(llr_2) * min(abs(llr_1), abs(llr_2))
    t1 = np.tanh(llr_1 / 2.0)
    t2 = np.tanh(llr_2 / 2.0)
    prod = np.clip(t1 * t2, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(prod)


def _slow_llr(i, N, llr, u_est):
    """递归计算第 i 个比特信道 LLR（Arikan 结构）"""
    if len(llr) != N:
        raise ValueError("llr length mismatch")
    if len(u_est) != i:
        raise ValueError("u_est length mismatch")

    if i == 0 and N == 1:
        return float(llr[0])

    if i % 2 == 0:
        llr_1 = _slow_llr(
            i // 2,
            N // 2,
            llr[: N // 2],
            (u_est[::2] ^ u_est[1::2])[: i // 2],
        )
        llr_2 = _slow_llr(
            i // 2,
            N // 2,
            llr[N // 2 :],
            u_est[1::2][: i // 2],
        )
        return _llr_check_node_operation(llr_1, llr_2)

    llr_1 = _slow_llr(
        (i - 1) // 2,
        N // 2,
        llr[: N // 2],
        (u_est[:-1:2] ^ u_est[1:-1:2])[: (i - 1) // 2],
    )
    llr_2 = _slow_llr(
        (i - 1) // 2,
        N // 2,
        llr[N // 2 :],
        u_est[1::2][: ((i - 1) // 2)],
    )
    return llr_2 + ((-1) ** u_est[-1]) * llr_1


class _FastSCContext:
    """O(N log N) SC LLR 计算上下文"""

    def __init__(self, N):
        self.N = N
        self.n = int(math.log2(N))
        self.llr_array = np.full(N * (self.n + 1), np.nan, dtype=np.float64)
        self.is_calc = [False] * (N * (self.n + 1))

    def reset(self):
        self.llr_array.fill(np.nan)
        self.is_calc = [False] * len(self.is_calc)

    def _problem_i(self, i):
        slice_idx = i // self.N
        modulus = 2 ** (self.n - slice_idx)
        return i % modulus

    def _descendants(self, i):
        slice_idx = i // self.N
        slice_i = i - slice_idx * self.N
        sub_len = 2 ** (self.n - slice_idx)
        sub_start = (slice_i // sub_len) * sub_len
        sub_i = i % sub_len
        left = (slice_idx + 1) * self.N + sub_start + (sub_i // 2)
        right = left + 2 ** (self.n - slice_idx - 1)
        return left, right

    def fast_llr(self, i, y, u_est):
        if not self.is_calc[i]:
            n = len(y)
            pi = self._problem_i(i)
            if pi == 0 and n == 1:
                self.llr_array[i] = y[0]
            else:
                if len(u_est) != pi:
                    raise ValueError("u_est length mismatch in fast_llr")
                h = n // 2
                left, right = self._descendants(i)
                if pi % 2 == 0:
                    l1 = self.fast_llr(left, y[:h], (u_est[::2] ^ u_est[1::2])[: pi // 2])
                    l2 = self.fast_llr(right, y[h:], u_est[1::2][: pi // 2])
                    self.llr_array[i] = _llr_check_node_operation(l1, l2)
                else:
                    l1 = self.fast_llr(
                        left, y[:h], (u_est[:-1:2] ^ u_est[1:-1:2])[: (pi - 1) // 2]
                    )
                    l2 = self.fast_llr(right, y[h:], u_est[1::2][: (pi - 1) // 2])
                    self.llr_array[i] = l2 + ((-1) ** u_est[-1]) * l1
            self.is_calc[i] = True
        return self.llr_array[i]


def sc_decode(llr_ch, frozen_bits):
    """SC 译码（主实现，O(N log N)）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=np.int8)
    ctx = _FastSCContext(N)
    frozen_set = set(np.where(frozen_bits)[0])

    for idx in range(N):
        if idx in frozen_set:
            u_hat[idx] = 0
            continue
        llr = ctx.fast_llr(idx, llr_ch, u_hat[:idx])
        u_hat[idx] = 0 if llr >= 0 else 1

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        p = phi
        for layer in range(n):
            if (p % 2) == 0:
                llr_layer_vec[phi].append(layer)
            else:
                bit_layer_vec[phi].append(layer)
            p //= 2
    return lambda_offset, llr_layer_vec, bit_layer_vec
