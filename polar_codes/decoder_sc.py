"""
极化码 SC（串行抵消）译码器
置换 SC（非递归，log-domain f 函数）
"""
import numpy as np
from encoder import _bitrev_indices


def _bitrev_scalar(x, n):
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


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(l1, l2):
    """log-domain box-plus（与 min-sum 兼容接口）"""
    if np.isscalar(l1) and np.isscalar(l2):
        if np.isinf(l1) and not np.isinf(l2):
            return l2
        if not np.isinf(l1) and np.isinf(l2):
            return l1
        if np.isinf(l1) and np.isinf(l2):
            return np.inf
        return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)
    l1 = np.asarray(l1, dtype=np.float64)
    l2 = np.asarray(l2, dtype=np.float64)
    out = np.empty_like(l1)
    for idx in np.ndindex(l1.shape):
        out[idx] = f_operation(l1[idx], l2[idx])
    return out


def g_operation(l1, l2, u_hat):
    u_hat = np.asarray(u_hat)
    if u_hat.ndim == 0 or u_hat.size == 1:
        b = int(u_hat) if u_hat.ndim else int(u_hat.item())
        if b == 0:
            if np.isinf(l1) or np.isinf(l2):
                return np.inf
            return l1 + l2
        return l1 - l2
    return np.where(
        u_hat == 0,
        np.where(np.isinf(l1) | np.isinf(l2), np.inf, l1 + l2),
        l1 - l2,
    )


def _frozen_set_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int) > 0)[0])


class _SCDState:
    def __init__(self, N, n, llr_ch):
        self.N = N
        self.n = n
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.L[:, 0] = llr_ch
        self.pm = 0.0

    def copy(self):
        st = _SCDState(self.N, self.n, self.L[:, 0])
        st.L[:] = self.L
        st.B[:] = self.B
        st.pm = self.pm
        return st

    def update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = 0 if np.isnan(self.B[j - branch_size, s + 1]) else self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = g_operation(self.L[j, s], self.L[j - branch_size, s], top_bit)

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    bj = 0 if np.isnan(self.B[j, s]) else int(self.B[j, s])
                    bt = 0 if np.isnan(self.B[j - branch_size, s]) else int(self.B[j - branch_size, s])
                    self.B[j - branch_size, s - 1] = bj ^ bt
                    self.B[j, s - 1] = bj


def sc_decode(llr_ch, frozen_bits):
    llr_ch = channel_llr_to_decoder(llr_ch)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = _frozen_set_from_mask(frozen_bits)
    state = _SCDState(N, n, llr_ch)

    for i in range(N):
        l = _bitrev_scalar(i, n)
        state.update_llrs(l)
        if l in frozen_set:
            state.B[l, n] = 0
        else:
            state.B[l, n] = 0 if state.L[l, n] >= 0 else 1
        state.update_bits(l)

    return state.B[:, n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = _bitrev_scalar(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def channel_llr_to_decoder(llr_ch):
    """极化编码含 B_N 时，将信道 LLR 映射到 SC 因子图顺序"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = _bitrev_indices(N)
    inv = np.argsort(br)
    return llr_ch[inv]


def sc_decode_channel(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
