"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x >= y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr_exact(l1, l2):
    """精确 log-domain f 函数（高 SNR 下与 min-sum 等价）"""
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr_exact(l1, l2, bit):
    if bit == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _prepare_channel_llr(llr_ch):
    """返回信道 LLR（自然顺序，与编码器一致）。"""
    return np.asarray(llr_ch, dtype=np.float64)


def _frozen_indices(frozen_bits):
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        return set(np.where(frozen_bits)[0])
    return set(np.where(frozen_bits.astype(int) == 1)[0])


def _update_llrs_at_index(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = _upper_llr_exact(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = _lower_llr_exact(L[j, s], L[j - branch_size, s], top_bit)


def _update_bits_at_index(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，与非递归 sc_decode 算法等价）。
    参数：
        llr: 长度 N 的信道 LLR 数组
        frozen_bits: 长度 N 的 bool/int 数组，1 表示冻结位（置 0）
    返回：
        u_hat: 长度 N 的估计源序列
    """
    llr = _prepare_channel_llr(llr)
    frozen = _frozen_indices(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    def decode_bit(i):
        if i >= N:
            return
        l = int(format(i, f"0{n}b")[::-1], 2)
        _update_llrs_at_index(L, B, l, n, N)
        if l in frozen:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            u_hat[l] = bit
            B[l, n] = bit
        _update_bits_at_index(B, l, n, N)
        decode_bit(i + 1)

    decode_bit(0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量：
      - lambda_offset[phi]: 第 phi 个比特对应的 LLR 存储偏移
      - llr_layer_vec[phi]: 第 phi 个比特需要执行 LLR 运算的层列表
      - bit_layer_vec[phi]: 第 phi 个比特需要执行比特返回的层列表
    """
    n = int(math.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        for layer in range(n):
            if ((phi >> layer) & 1) == 0:
                layers.append(layer)
        llr_layer_vec.append(layers)

        bit_layers = []
        temp = phi
        for layer in range(n):
            if temp % 2 == 1:
                bit_layers.append(layer)
            temp //= 2
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Vangala 2014 高效实现）。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（float64）
        frozen_bits: 长度 N 的 bool/int 数组，1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列（0/1 int 数组）
    """
    llr = _prepare_channel_llr(llr_ch)
    frozen = _frozen_indices(frozen_bits)
    N = len(llr)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        l = int(format(i, f"0{n}b")[::-1], 2)
        _update_llrs_at_index(L, B, l, n, N)
        if l in frozen:
            u_hat[l] = 0
            B[l, n] = 0
        else:
            bit = 0 if L[l, n] >= 0 else 1
            u_hat[l] = bit
            B[l, n] = bit
        _update_bits_at_index(B, l, n, N)

    return u_hat
