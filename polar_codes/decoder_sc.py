"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversed_index


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """找到索引 i 的二进制表示中第一个 1 的位置（从高位计）"""
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
    """找到索引 i 的二进制表示中第一个 0 的位置（从高位计）"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def polar_dec(L, level, idx):
        if level == 0:
            phi = idx
            if frozen_bits[phi]:
                u_hat[phi] = 0
            else:
                u_hat[phi] = 0 if L[0] >= 0 else 1
            return
        half = 1 << (level - 1)
        polar_dec(f_operation(L[:half], L[half:]), level - 1, idx)
        polar_dec(
            g_operation(L[:half], L[half:], u_hat[idx : idx + half]),
            level - 1,
            idx + half,
        )

    polar_dec(llr, n, 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        br_phi = bit_reversed_index(phi, n)
        start = n - _active_llr_level(br_phi, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - _active_bit_level(br_phi, n) + 1
        bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（层序更新，按比特倒序译码）。
    信道 LLR 按码字自然顺序输入，内部自动做比特倒序以匹配编码器。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    # 信道 LLR 倒序以匹配极化因子图
    rev = np.array([bit_reversed_index(i, n) for i in range(N)], dtype=int)
    llr_ch = llr_ch[rev]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch

    decode_order = [bit_reversed_index(i, n) for i in range(N)]

    for l in decode_order:
        # 自底向上更新 LLR
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    top_llr = L[j, s]
                    btm_llr = L[j + branch_size, s]
                    L[j, s + 1] = f_operation(top_llr, btm_llr)
                else:
                    btm_llr = L[j, s]
                    top_llr = L[j - branch_size, s]
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        # 比特回传
        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(
                            B[j - branch_size, s]
                        )
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
