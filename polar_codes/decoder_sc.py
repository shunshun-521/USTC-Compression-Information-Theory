"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


from encoder import bit_reversal_permutation


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """SC 译码 f 运算：精确 box-plus"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    result = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    it = np.nditer([La, Lb, result], flags=['refs_ok'], op_flags=[['readonly'], ['readonly'], ['writeonly']])
    for la, lb, out in it:
        a, b = float(la), float(lb)
        out[...] = _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)
    return result


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed(x, n):
    """比特倒序索引"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def active_llr_level(i, n):
    """LLR 更新起始层"""
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
    """比特回传起始层"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = llr[bit_reversal_permutation(len(llr))]
    u_hat = np.zeros(len(llr), dtype=int)

    def decode_rec(llr_node, frozen_node, offset):
        m = len(llr_node)
        if m == 1:
            if frozen_node[0]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_node[0] >= 0 else 1
            return
        half = m // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_rec(llr_left, frozen_node[:half], offset)
        u_left = u_hat[offset:offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_rec(llr_right, frozen_node[half:], offset + half)

    decode_rec(llr, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << (n - i) for i in range(n + 1)]
    llr_layer_vec = [list(range(n - 1, -1, -1)) for _ in range(N)]
    bit_layer_vec = []
    for phi in range(N):
        blayers = []
        for l in range(n):
            blayers.append(l)
            if (phi >> l) & 1:
                break
        bit_layer_vec.append(blayers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    按比特倒序索引顺序译码，与标准极化码蝶形编码匹配。
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    rev = bit_reversal_permutation(N)
    llr_ch = llr_ch[rev]

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_ch

    for idx in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(idx, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if frozen_bits[idx]:
            B[idx, n] = 0
        else:
            B[idx, n] = 0 if L[idx, n] >= 0 else 1

        if idx >= N // 2:
            for s in range(n, n - active_bit_level(idx, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(idx, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)
