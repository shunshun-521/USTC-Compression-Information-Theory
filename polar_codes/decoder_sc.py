"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
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
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed(i, n):
    result = 0
    for bit in range(n):
        if i & (1 << bit):
            result |= 1 << (n - 1 - bit)
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


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    参数：
        llr: 长度 N 的信道 LLR 数组
        frozen_bits: 长度 N 的 bool 数组，True 表示冻结位（置 0）
    返回：
        u_hat: 长度 N 的估计源序列
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, depth, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, depth - 1, bit_offset)

        u_left = u_hat[bit_offset:bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, depth - 1, bit_offset + half)

    decode_node(llr, int(math.log2(N)), 0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回 decode_order, n（按比特倒序的译码顺序与层数）。
    """
    n = int(math.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    return decode_order, n


def _update_llrs(L, B, l, n):
  for s in range(n - _active_llr_level(l, n), n):
    block_size = 2 ** (s + 1)
    branch_size = block_size // 2
    for j in range(l, len(L), block_size):
      if j % block_size < branch_size:
        L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
      else:
        L[j, s + 1] = g_operation(
            L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
        )


def _update_bits(B, l, n):
  if l < len(B) / 2:
    return
  for s in range(n, n - _active_bit_level(l, n), -1):
    block_size = 2 ** s
    branch_size = block_size // 2
    for j in range(l, -1, -block_size):
      if j % block_size >= branch_size:
        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
        B[j, s - 1] = B[j, s]


def _sc_decode_core(llr_ch, frozen_bits):
    """非递归 SC 译码核心。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    frozen_set = set(np.where(frozen_bits)[0])

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    信道 LLR 直接对应编码输出顺序（含比特倒序置换后的码字）。
    """
    return _sc_decode_core(llr_ch, frozen_bits)
