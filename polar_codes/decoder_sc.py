"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：u_hat 为左子树对应的部分和比特"""
    return np.where(u_hat, Lb - La, La + Lb)


def _bit_reversed(i, n):
    b = format(i, f"0{n}b")[::-1]
    return int(b, 2)


def _active_llr_level(l, n):
    tmp = l
    level = 0
    while tmp % 2 == 1:
        level += 1
        tmp >>= 1
    return level


def _active_bit_level(l, n):
    if l < 2 ** (n - 1):
        return 0
    tmp = l
    level = 0
    while tmp % 2 == 0:
        level += 1
        tmp >>= 1
    return level


def _upper_llr(top, btm):
    return f_operation(top, btm)


def _lower_llr(btm, top, top_bit):
    return g_operation(top, btm, top_bit)


# ==================== 递归 SC 译码（参考实现）====================


def _sc_decode_core(llr, frozen_bits):
    """递归 SC 核心（输入 LLR 已为译码器顺序）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    decisions = [0] * N

    def _sc(llrs, indices):
        if len(indices) == 1:
            i = indices[0]
            if frozen_bits[i]:
                decisions[i] = 0
            else:
                decisions[i] = 0 if llrs[0] >= 0 else 1
            return [decisions[i]]

        half = len(indices) // 2
        left_llr = [_upper_llr(llrs[i], llrs[i + half]) for i in range(half)]
        enc_left = _sc(left_llr, indices[:half])
        right_llr = [_lower_llr(llrs[i + half], llrs[i], enc_left[i]) for i in range(half)]
        enc_right = _sc(right_llr, indices[half:])
        return [enc_left[i] ^ enc_right[i] for i in range(half)] + enc_right

    _sc(llr.tolist(), list(range(N)))
    return np.array(decisions, dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 dit.coding.polar 一致）"""
    from channel import reorder_llr_for_decoder

    llr = np.asarray(llr, dtype=np.float64)
    llr = reorder_llr_for_decoder(llr, llr.size)
    return _sc_decode_core(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


class _SCD:
    """Permuted successive cancellation decoder (Vangala et al.)"""

    def __init__(self, N, frozen_bits):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen = set(np.where(frozen_bits)[0])
        self.L = np.full((N, self.n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, self.n + 1), dtype=np.int8)

    def decode(self, llr_ch):
        self.L[:, self.n] = llr_ch
        for l in [_bit_reversed(i, self.n) for i in range(self.N)]:
            self._update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = 0 if self.L[l, self.n] >= 0 else 1
            self._update_bits(l)
        return self.B[:, self.n].astype(int)

    def _update_llrs(self, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = _upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)

    def _update_bits(self, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = (self.B[j, s] ^ self.B[j - branch_size, s]) % 2
                    self.B[j, s - 1] = self.B[j, s]


_SCD_CACHE = {}


def precompute_sc_indices(N):
    """兼容接口：返回占位表（非递归实现使用 SCD 类）"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def _get_sc_tables(N):
    if N not in _SCD_CACHE:
        _SCD_CACHE[N] = precompute_sc_indices(N)
    return _SCD_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数（递归实现，保证与大码长一致）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
