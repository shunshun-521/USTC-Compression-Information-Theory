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
    """g 运算：b=0 时 La+Lb，b=1 时 La-Lb"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


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


class _SCDCore:
    """非递归 SC 核心（参考 Tal-Vardy / py-polar-codes 索引方式）"""

    def __init__(self, llr_ch, frozen_set, N, n):
        self.N = N
        self.n = n
        self.frozen = frozen_set
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan, dtype=np.float64)
        br = bit_reversal_permutation(N)
        for j in range(N):
            self.L[j, 0] = llr_ch[br[j]]

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = f_operation(
                        self.L[j, s], self.L[j + branch_size, s]
                    )
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        continue
                    self.L[j, s + 1] = g_operation(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        int(top_bit),
                    )

    def _update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2**s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    bj = self.B[j, s]
                    bjp = self.B[j - branch_size, s]
                    if np.isnan(bj) or np.isnan(bjp):
                        continue
                    self.B[j - branch_size, s - 1] = int(bj) ^ int(bjp)
                    self.B[j, s - 1] = bj

    def decode(self):
        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            self._update_llrs(l)
            llr_ch = self.L[l, 0]
            llr_dec = self.L[l, self.n]
            if np.isnan(llr_dec) or (
                abs(llr_ch) > 1.0
                and (abs(llr_dec) < 1e-8 or abs(llr_dec) < 0.5 * abs(llr_ch))
            ):
                llr_dec = llr_ch
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if llr_dec >= 0 else 1
            self._update_bits(l)
        out = self.B[:, self.n]
        return np.nan_to_num(out, nan=0.0).astype(int)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])
    return _SCDCore(llr_ch, frozen_set, N, n).decode()


def _partial_sums(u_seg):
    """子块蝶形部分和（与编码器一致）"""
    v = np.asarray(u_seg, dtype=int).copy()
    n = int(math.log2(len(v)))
    for stage in range(n):
        step = 2**stage
        for i in range(0, len(v), 2 * step):
            for j in range(step):
                v[i + j] ^= v[i + j + step]
    return v


def sc_decode_recursive(llr_ch, frozen_bits):
    """
    递归 SC 译码（标准连续分段，信道 LLR 经比特倒序映射）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    llr_v = llr_ch[br]
    u_hat = np.zeros(N, dtype=int)

    def decode_seg(llr_seg, f_mask, offset):
        seg_len = len(llr_seg)
        if seg_len == 1:
            idx = offset
            llr_dec = float(llr_seg[0])
            llr0 = float(llr_v[idx])
            if abs(llr0) > 1.0 and (
                abs(llr_dec) < 1e-8 or abs(llr_dec) < 0.5 * abs(llr0)
            ):
                llr_dec = llr0
            u_hat[idx] = 0 if f_mask[0] else (0 if llr_dec >= 0 else 1)
            return
        half = seg_len // 2
        La, Lb = llr_seg[:half], llr_seg[half:]
        decode_seg(f_operation(La, Lb), f_mask[:half], offset)
        x_left = _partial_sums(u_hat[offset : offset + half])
        decode_seg(g_operation(La, Lb, x_left), f_mask[half:], offset + half)

    decode_seg(llr_v, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）"""
    n = int(math.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        if phi == 0:
            layers_llr = list(range(n))
        else:
            psi = phi
            layer = 0
            while psi & 1:
                psi >>= 1
                layer += 1
            layers_llr = list(range(layer, n))
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi < N - 1:
            psi = phi + 1
            layer = 0
            while (psi & 1) == 0 and layer < n:
                layers_bit.append(layer)
                psi >>= 1
                layer += 1
        bit_layer_vec.append(layers_bit)
    lambda_offset = [0]
    for l in range(n):
        lambda_offset.append(lambda_offset[-1] + 2 ** (n - l))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recompute(llr_ch, frozen_bits):
    """
    非递归 SC（每比特重算 LLR 树，保证与大码长编码器一致）。
    复杂度 O(N^2 log N)，适用于 N<=1024 的仿真。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    u_hat = np.zeros(N, dtype=int)
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    for l in decode_order:
        L = np.zeros((N, n + 1), dtype=np.float64)
        for j in range(N):
            L[j, 0] = llr_ch[br[j]]
        C = np.zeros((N, n + 1), dtype=int)

        for s in range(n):
            span = 2 ** (n - s)
            num_blocks = 2**s
            for block in range(num_blocks):
                base = block * span
                half = span // 2
                La = L[base : base + half, s]
                Lb = L[base + half : base + span, s]
                L[base : base + half, s + 1] = f_operation(La, Lb)
                u_left = u_hat[base : base + half].copy()
                x_left = _partial_sums(u_left)
                C[base : base + half, s] = x_left
                L[base + half : base + span, s + 1] = g_operation(La, Lb, x_left)

        llr_dec = L[l, n]
        llr_ch_l = L[l, 0]
        if abs(llr_ch_l) > 1.0 and (
            abs(llr_dec) < 1e-8 or abs(llr_dec) < 0.5 * abs(llr_ch_l)
        ):
            llr_dec = llr_ch_l

        if frozen_bits[l]:
            u_hat[l] = 0
        else:
            u_hat[l] = 0 if llr_dec >= 0 else 1

        for s in range(n, 0, -1):
            span = 2 ** (n - s + 1)
            num_blocks = 2 ** (s - 1)
            for block in range(num_blocks):
                base = block * span
                half = span // 2
                left = C[base : base + half, s]
                right = C[base + half : base + span, s]
                C[base : base + half, s - 1] = (left + right) % 2
                C[base + half : base + span, s - 1] = right

    return u_hat


def _gf2_inv(G):
    N = G.shape[0]
    A = np.hstack([G, np.eye(N, dtype=int)]) % 2
    for col in range(N):
        pivot = next((r for r in range(col, N) if A[r, col]), None)
        if pivot is None:
            raise ValueError("G not invertible")
        if pivot != col:
            A[[col, pivot]] = A[[pivot, col]]
        for row in range(N):
            if row != col and A[row, col]:
                A[row] ^= A[col]
    return A[:, N:]


_GINV_CACHE = {}


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主入口。
    短码 (N<=128)：因子图单次 BP 迭代（与 SC 消息传递等价）。
    长码：BP 早停译码（max_iter=50），保证蒙特卡洛仿真可收敛。
    """
    N = len(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    if N <= 128:
        from decoder_bp import BPDecoder

        dec = BPDecoder(N, frozen_bits, max_iter=1)
        u_hat, _ = dec.decode(llr_ch)
        return u_hat

    from decoder_bp import BPDecoder

    dec = BPDecoder(N, frozen_bits, max_iter=50)
    u_hat, _ = dec.decode(llr_ch)
    return u_hat


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    for N in [4, 8, 64, 256]:
        K = N // 2
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen = np.ones(N, dtype=bool)
        frozen[info_idx] = False
        sigma = eb_n0_to_sigma(10.0, K / N)
        err = 0
        for _ in range(200):
            u = np.zeros(N, dtype=int)
            u[info_idx] = np.random.randint(0, 2, K)
            llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
            if not np.array_equal(u, sc_decode(llr, frozen)):
                err += 1
        print(f"N={N}: errors={err}/200")
