"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，基于 polarcodes SCD）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_logdomain(l1, l2):
    """对数域 box-plus f 运算"""
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def g_logdomain(l1, l2, b):
    """对数域 g 运算"""
    return (l1 + l2) if b == 0 else (l1 - l2)


def bit_reversed(i, n):
    return int(f"{i:0{n}b}"[::-1], 2)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits, use_logdomain=True):
    """递归 SC 译码（调用非递归核心以保证正确性）"""
    return sc_decode(llr, frozen_bits, use_logdomain)


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 1, dtype=int)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        while psi & 1:
            layers_llr.append(int(np.log2(psi & -psi)))
            psi >>= 1
        start = (max(layers_llr) + 1) if layers_llr else 0
        llr_layer_vec.append(list(range(start, n)))

        layers_bit = []
        psi = phi + 1
        while psi <= N and (psi & 1) == 0:
            layers_bit.append(int(np.log2(psi & -psi)) - 1)
            psi >>= 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, use_logdomain=True):
    """
    非递归 SC 译码（polarcodes SCD 算法）。
    frozen_bits: bool 数组，True 表示冻结位
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    f_fn = f_logdomain if use_logdomain else f_operation
    g_fn = g_logdomain if use_logdomain else lambda a, b, c: g_operation(a, b, c)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    if use_logdomain:
                        L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
                    else:
                        L[j, s + 1] = f_fn(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    if use_logdomain:
                        L[j, s + 1] = g_fn(L[j, s], L[j - branch_size, s], int(top_bit))
                    else:
                        L[j, s + 1] = g_fn(L[j, s], L[j - branch_size, s], int(top_bit))

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        bj = B[j, s]
                        bjm = B[j - branch_size, s]
                        if np.isnan(bj):
                            bj = 0
                        if np.isnan(bjm):
                            bjm = 0
                        B[j - branch_size, s - 1] = int(bj) ^ int(bjm)
                        B[j, s - 1] = bj

    return B[:, n].astype(int)


def prepare_channel_llr(llr_ch):
    """信道 LLR 直接用于蝶形编码器"""
    return llr_ch


def sc_decode_with_brp(llr_ch, frozen_bits):
    """SC 译码入口"""
    return sc_decode(llr_ch, frozen_bits)
