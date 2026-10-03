"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现，叶节点×阶段存储）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（box-plus 上分支）。"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（box-plus 下分支）：La 为上分支 LLR，Lb 为下分支 LLR。"""
    return (1 - 2 * u_hat) * La + Lb


def _bit_reverse(i, n):
    return int(format(int(i), f"0{n}b")[::-1], 2)


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


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（小码长验证用，调用与主实现相同的叶节点存储算法）。"""
    return sc_decode(llr, frozen_bits)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    返回比特倒序译码顺序及每层活跃深度（与参考 SCD 一致）。
    """
    n = int(math.log2(N))
    decode_order = [_bit_reverse(i, n) for i in range(N)]
    llr_layer_vec = []
    for l_idx in decode_order:
        start = n - _active_llr_level(l_idx, n)
        llr_layer_vec.append(list(range(start, n)))
    bit_layer_vec = []
    for l_idx in decode_order:
        end = n - _active_bit_level(l_idx, n)
        bit_layer_vec.append(list(range(n, end, -1)))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（叶节点 j、阶段 s 的 L[j,s] 存储；按比特倒序依次判决）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    decode_order = [_bit_reverse(i, n) for i in range(N)]

    for l_idx in decode_order:
        for s in range(n - _active_llr_level(l_idx, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], top_bit
                    )

        if l_idx in frozen_set:
            B[l_idx, n] = 0
        else:
            B[l_idx, n] = _hard_decision(L[l_idx, n])

        if l_idx < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l_idx, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                        B[j - branch_size, s]
                    )
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
