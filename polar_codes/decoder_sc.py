"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


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
    return (1 - 2 * u_hat) * La + Lb


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _f_exact(l1, l2):
  if np.isinf(l1) and not np.isinf(l2):
      return l2
  if not np.isinf(l1) and np.isinf(l2):
      return l1
  if np.isinf(l1) and np.isinf(l2):
      return np.inf
  return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _g_exact(l_bottom, l_top, bit):
    if bit == 0:
        if np.isinf(l_bottom) or np.isinf(l_top):
            return np.inf
        return l_bottom + l_top
    return l_bottom - l_top


def _bit_reversed_index(i, n):
    result = 0
    for j in range(n):
        if i & (1 << j):
            result |= 1 << (n - 1 - j)
    return result


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


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    if 2 ** n != N:
        raise ValueError("N must be a power of 2")

    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers_llr = []
        psi = phi
        layer = 0
        while layer < n:
            if (psi & 1) == 0:
                layers_llr.append(layer)
            psi >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        psi = phi
        layer = 0
        while layer < n:
            if (psi & 1) == 1:
                layers_bit.append(layer)
            psi >>= 1
            layer += 1
        if phi == N - 1:
            for l in range(n):
                if l not in layers_bit:
                    layers_bit.append(l)
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，使用精确 log-domain f 运算）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    llr = llr[bit_reversal_permutation(N)]
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        length = len(llr_node)
        if length == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = length // 2
        llr_left = np.array(
            [_f_exact(llr_node[i], llr_node[i + half]) for i in range(half)],
            dtype=np.float64,
        )
        decode_node(llr_left, bit_offset)

        llr_right = np.array(
            [
                _g_exact(llr_node[i + half], llr_node[i], u_hat[bit_offset + i])
                for i in range(half)
            ],
            dtype=np.float64,
        )
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


def sc_decode(llr_ch, frozen_bits, use_min_sum=False):
    """
    非递归 SC 译码主函数（高效实现）。

    信道 LLR 按编码输出顺序输入，内部做比特倒序后与编码器蝶形结构对齐。
  译码顺序为 bit-reversed 自然序。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    llr_ch = llr_ch[bit_reversal_permutation(N)]
    frozen_set = set(np.where(frozen_bits)[0])

    f_func = f_operation if use_min_sum else _f_exact
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    def update_llrs(l):
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_func(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = _g_exact(
                        L[j, s],
                        L[j - branch_size, s],
                        B[j - branch_size, s + 1],
                    )

    def update_bits(l):
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    for i in range(N):
        l = _bit_reversed_index(i, n)
        update_llrs(l)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = int(B[l, n])
        update_bits(l)

    return u_hat
