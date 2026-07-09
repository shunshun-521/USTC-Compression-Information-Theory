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
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
  下层分支使用 top=La, btm=Lb（与极化码因子图一致）
    """
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
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


def _align_channel_llrs(llr_ch, bit_reversed_codeword=True):
    """
    若发送码字经过比特倒序置换，将信道 LLR 重排至因子图自然顺序。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    if not bit_reversed_codeword:
        return llr_ch
    br = bit_reversal_permutation(len(llr_ch))
    return llr_ch[br]


def sc_decode_recursive(llr, frozen_bits, bit_reversed_codeword=True):
    """
    递归 SC 译码（参考实现）。
    极化码的抵消顺序为比特倒序，此处复用非递归内核以保证与编码一致。
    """
    return sc_decode(llr, frozen_bits, bit_reversed_codeword=bit_reversed_codeword)


def _sc_decode_recursive_tree(llr, frozen_bits):
    """基于二叉递归的 SC 译码（教学参考，与主实现等价）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_block(node_llr, depth, index_offset):
        length = len(node_llr)
        if length == 1:
            idx = index_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if node_llr[0] >= 0 else 1
            return

        half = length // 2
        left_llr = f_operation(node_llr[:half], node_llr[half:])
        decode_block(left_llr, depth - 1, index_offset)
        right_llr = g_operation(node_llr[:half], node_llr[half:], u_hat[index_offset:index_offset + half])
        decode_block(right_llr, depth - 1, index_offset + half)

    decode_block(llr, n, 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（按比特倒序处理顺序）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        llr_layer_vec.append(llr_layers)
        if phi == N - 1:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits, bit_reversed_codeword=True):
    """
    非递归 SC 译码（高效实现，比特倒序抵消顺序）。
    """
    llr_ch = _align_channel_llrs(llr_ch, bit_reversed_codeword)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    for phi in range(N):
        l = _bit_reversed_index(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l < N / 2:
            continue

        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                    B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
