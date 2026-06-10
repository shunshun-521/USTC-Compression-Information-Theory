"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

CHECK_NODE_TANH_THRES = 44


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（box-plus）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  支持向量化（La, Lb 为同形状 numpy 数组）
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.shape != () and La.shape == Lb.shape:
        return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
    if abs(La) > CHECK_NODE_TANH_THRES and abs(Lb) > CHECK_NODE_TANH_THRES:
        if La * Lb > 0:
            return min(abs(La), abs(Lb))
        return -min(abs(La), abs(Lb))
    ta = np.tanh(La / 2.0)
    tb = np.tanh(Lb / 2.0)
    prod = np.clip(ta * tb, -1.0 + 1e-15, 1.0 - 1e-15)
    return float(2.0 * np.arctanh(prod))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La + Lb


def _slow_llr_recursive(i, N, llr_vec, u_est):
    """按 Arikan 偶/奇结构递归计算第 i 个比特的 LLR。"""
    if i == 0 and N == 1:
        return float(llr_vec[0])

    if i % 2 == 0:
        llr_1 = _slow_llr_recursive(
            i // 2,
            N // 2,
            llr_vec[: N // 2],
            (u_est[::2] ^ u_est[1::2])[: i // 2],
        )
        llr_2 = _slow_llr_recursive(
            i // 2,
            N // 2,
            llr_vec[N // 2 :],
            u_est[1::2][: i // 2],
        )
        return f_operation(llr_1, llr_2)

    llr_1 = _slow_llr_recursive(
        (i - 1) // 2,
        N // 2,
        llr_vec[: N // 2],
        (u_est[:-1:2] ^ u_est[1:-1:2])[:(i - 1) // 2],
    )
    llr_2 = _slow_llr_recursive(
        (i - 1) // 2,
        N // 2,
        llr_vec[N // 2 :],
        u_est[1::2][: ((i - 1) // 2)],
    )
    return llr_2 + ((-1) ** u_est[-1]) * llr_1


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
        else:
            llr_i = _slow_llr_recursive(i, N, llr, u_hat[:i])
            u_hat[i] = 0 if llr_i >= 0 else 1

    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助结构（兼容接口）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [[] for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    for phi in range(N):
        psi = phi
        layer = 0
        while psi & 1:
            llr_layer_vec[phi].append(layer)
            psi >>= 1
            layer += 1
        if (phi & 1) == 0 and phi > 0:
            psi = phi
            layer = 0
            while (psi & 1) == 0:
                bit_layer_vec[phi].append(layer)
                psi >>= 1
                layer += 1
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _FastLLRCache:
    """非递归 SC 的 LLR 缓存（O(N log N)）。"""

    def __init__(self, N, llr_ch):
        self.N = N
        self.n = int(math.log2(N))
        self.llr_ch = np.asarray(llr_ch, dtype=np.float64)
        self.llr_array = np.zeros(N * (self.n + 1), dtype=np.float64)
        self.is_calc = [False] * (N * (self.n + 1))

    def _problem_i(self, idx):
        slice_idx = idx // self.N
        modulus = 1 << (self.n - slice_idx)
        return idx % modulus

    def _descendants(self, idx):
        slice_idx = idx // self.N
        slice_i = idx - slice_idx * self.N
        sub_len = 1 << (self.n - slice_idx)
        sub_start = (slice_i // sub_len) * sub_len
        sub_i = idx % sub_len
        left = (slice_idx + 1) * self.N + sub_start + (sub_i // 2)
        right = left + (1 << (self.n - slice_idx - 1))
        return left, right

    def copy(self):
        cloned = _FastLLRCache.__new__(_FastLLRCache)
        cloned.N = self.N
        cloned.n = self.n
        cloned.llr_ch = self.llr_ch
        cloned.llr_array = self.llr_array.copy()
        cloned.is_calc = self.is_calc.copy()
        return cloned

    def get_llr(self, bit_index, u_est):
        return self._fast_llr(bit_index, self.llr_ch, u_est)

    def _fast_llr(self, idx, llr_vec, u_est):
        if self.is_calc[idx]:
            return self.llr_array[idx]

        problem_i = self._problem_i(idx)
        N = len(llr_vec)

        if problem_i == 0 and N == 1:
            self.llr_array[idx] = float(llr_vec[0])
        else:
            left_desc, right_desc = self._descendants(idx)
            if problem_i % 2 == 0:
                llr_1 = self._fast_llr(
                    left_desc,
                    llr_vec[: N // 2],
                    (u_est[::2] ^ u_est[1::2])[: problem_i // 2],
                )
                llr_2 = self._fast_llr(
                    right_desc,
                    llr_vec[N // 2 :],
                    u_est[1::2][: problem_i // 2],
                )
                self.llr_array[idx] = f_operation(llr_1, llr_2)
            else:
                llr_1 = self._fast_llr(
                    left_desc,
                    llr_vec[: N // 2],
                    (u_est[:-1:2] ^ u_est[1:-1:2])[:(problem_i // 2)],
                )
                llr_2 = self._fast_llr(
                    right_desc,
                    llr_vec[N // 2 :],
                    u_est[1::2][: (problem_i // 2)],
                )
                self.llr_array[idx] = llr_2 + ((-1) ** u_est[-1]) * llr_1

        self.is_calc[idx] = True
        return self.llr_array[idx]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（LLR 缓存，O(N log N)）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    u_hat = np.zeros(N, dtype=int)
    cache = _FastLLRCache(N, llr_ch)

    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
        else:
            llr_i = cache.get_llr(i, u_hat[:i])
            u_hat[i] = 0 if llr_i >= 0 else 1

    return u_hat


def verify_sc_decoders(N=64, num_frames=100, eb_n0_db=10.0):
    """在极低噪声下验证递归与非递归 SC 译码一致性。"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode

    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

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
        u_ref = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(u_rec, u_ref):
            raise AssertionError("Non-recursive and recursive SC disagree")
        if not np.array_equal(u[info_idx], u_rec[info_idx]):
            raise AssertionError("SC decode error at high SNR")
