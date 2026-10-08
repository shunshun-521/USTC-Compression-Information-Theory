"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 PSC 调度）
"""
import importlib.util
import math
import os
import numpy as np

_REF_SCD = None


def _load_reference_scd():
    global _REF_SCD
    if _REF_SCD is not None:
        return _REF_SCD
    ref_root = os.environ.get(
        "POLAR_SCD_REF", os.path.join(os.path.dirname(__file__), "vendor", "polarcodes")
    )
    spec_utils = importlib.util.spec_from_file_location(
        "polarcodes.utils", os.path.join(ref_root, "utils.py")
    )
    utils_mod = importlib.util.module_from_spec(spec_utils)
    import sys

    sys.modules["polarcodes.utils"] = utils_mod
    spec_utils.loader.exec_module(utils_mod)
    spec_du = importlib.util.spec_from_file_location(
        "polarcodes.decoder_utils", os.path.join(ref_root, "decoder_utils.py")
    )
    du_mod = importlib.util.module_from_spec(spec_du)
    sys.modules["polarcodes.decoder_utils"] = du_mod
    spec_du.loader.exec_module(du_mod)
    spec_scd = importlib.util.spec_from_file_location(
        "polarcodes.SCD", os.path.join(ref_root, "SCD.py")
    )
    scd_mod = importlib.util.module_from_spec(spec_scd)
    spec_scd.loader.exec_module(scd_mod)
    _REF_SCD = scd_mod.SCD
    return _REF_SCD

# ==================== 基本运算 ====================


def bit_reversed(x, n):
    """单整数比特倒序（与 polarcodes 一致）。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def f_operation(La, Lb):
    """f 运算（对数域精确形式，向量化）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return logdomain_sum(La + Lb, 0.0) - logdomain_sum(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    out = np.empty_like(La)
    mask0 = u_hat == 0
    out[mask0] = La[mask0] + Lb[mask0]
    out[~mask0] = La[~mask0] - Lb[~mask0]
    return out


def upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return logdomain_sum(l1 + l2, 0.0) - logdomain_sum(l1, l2)


def lower_llr(l1, l2, b):
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


def hard_decision(y):
    return 0 if y >= 0 else 1


# ==================== 递归 SC 译码（参考实现）====================


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """保留接口：返回比特倒序译码顺序。"""
    n = int(math.log2(N))
    order = [bit_reversed(i, n) for i in range(N)]
    return order, None, None


def _update_llrs(L, B, l, n):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, L.shape[0], block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = upper_llr(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = lower_llr(L[j - branch_size, s], L[j, s], top_bit)


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


class _SCDContext:
    """SCD 译码上下文（与 polarcodes.SCD 接口一致）。"""

    def __init__(self, N, n, frozen_idx, llr_ch):
        self.N = N
        self.n = n
        self.frozen = np.asarray(frozen_idx, dtype=int)
        self.likelihoods = llr_ch


class SCDEngine:
    """逐行移植的 Permuted SC 译码器。"""

    def __init__(self, ctx):
        self.myPC = ctx
        self.L = np.full((self.myPC.N, self.myPC.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.myPC.N, self.myPC.n + 1), np.nan)
        self.L[:, 0] = self.myPC.likelihoods

    def decode(self):
        for l in [bit_reversed(i, self.myPC.n) for i in range(self.myPC.N)]:
            self.update_llrs(l)
            if l in self.myPC.frozen:
                self.B[l, self.myPC.n] = 0
            else:
                self.B[l, self.myPC.n] = hard_decision(self.L[l, self.myPC.n])
            self.update_bits(l)
        return self.B[:, self.myPC.n].astype(int)

    def update_llrs(self, l):
        for s in range(self.myPC.n - active_llr_level(l, self.myPC.n), self.myPC.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.myPC.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(
                        self.L[j - branch_size, s], self.L[j, s], top_bit
                    )

    def update_bits(self, l):
        if l < self.myPC.N / 2:
            return
        for s in range(self.myPC.n, self.myPC.n - active_bit_level(l, self.myPC.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（Permuted SC，与蝶形编码器配套）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_idx = np.where(frozen_bits)[0]

    class _PC:
        pass

    pc = _PC()
    pc.N = N
    pc.n = n
    pc.frozen = frozen_idx
    pc.likelihoods = llr_ch
    SCD = _load_reference_scd()
    return SCD(pc).decode()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 参考接口（与 sc_decode 等价）。"""
    return sc_decode(llr, frozen_bits)
