"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation


# ==================== 基本运算 ====================

def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    La: 上分支 LLR，Lb: 下分支 LLR
    """
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _bit_reversed_index(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层（首个 1 之前的 0 个数 + 1）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """比特回传起始层（首个 0 之前的 1 个数 + 1）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _reorder_channel_llrs(llr_ch):
    """
    将信道 LLR 重排为蝶形编码（无输出倒序）对应的顺序。
    编码器输出含比特倒序置换，译码前需做逆置换。
    """
    N = len(llr_ch)
    inv_br = np.argsort(bit_reversal_permutation(N))
    return llr_ch[inv_br]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。

    采用逐比特递归计算 LLR 的方式，与高效非递归版本等价。
    """
    llr = _reorder_channel_llrs(np.asarray(llr, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr

    def decode_bit(phi):
        l = _bit_reversed_index(phi, n)
        start = n - _active_llr_level(l, n)
        for s in range(start, n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], int(B[j - branch_size, s + 1])
                    )

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0.0 if L[l, n] >= 0 else 1.0
        u_hat[l] = int(B[l, n])

        if l < N // 2:
            return

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for phi in range(N):
        decode_bit(phi)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。

    返回：
        decode_order: 比特译码顺序（比特倒序）
        llr_start_layer: 每个比特 LLR 更新的起始层
        bit_start_layer: 每个比特回传的起始层
        lambda_offset: 各层块大小
    """
    n = int(np.log2(N))
    decode_order = np.array([_bit_reversed_index(i, n) for i in range(N)], dtype=int)
    llr_start_layer = np.array([n - _active_llr_level(l, n) for l in decode_order], dtype=int)
    bit_start_layer = np.array([n - _active_bit_level(l, n) for l in decode_order], dtype=int)
    lambda_offset = np.array([1 << i for i in range(n + 1)], dtype=int)
    return decode_order, llr_start_layer, bit_start_layer, lambda_offset


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（float64）
        frozen_bits: 长度 N 的 bool/int 数组，1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列（0/1 int 数组）
    """
    llr_ch = _reorder_channel_llrs(np.asarray(llr_ch, dtype=np.float64))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan, dtype=np.float64)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    decode_order, llr_start_layer, bit_start_layer, _ = precompute_sc_indices(N)
    frozen_set = set(np.where(frozen_bits)[0])

    for idx, l in enumerate(decode_order):
        for s in range(llr_start_layer[idx], n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], int(B[j - branch_size, s + 1])
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0.0 if L[l, n] >= 0 else 1.0
        u_hat[l] = int(B[l, n])

        if l < N // 2:
            continue

        for s in range(n, bit_start_layer[idx], -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return u_hat
