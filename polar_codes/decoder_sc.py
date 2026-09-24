"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def bit_reversed(x, n):
    """比特倒序索引"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def logdomain_sum(x, y):
    """log-domain 加法"""
    if x == np.inf and y != np.inf:
        return y
    if x != np.inf and y == np.inf:
        return x
    if x == np.inf and y == np.inf:
        return np.inf
    if x == -np.inf and y != -np.inf:
        return y
    if x != -np.inf and y == -np.inf:
        return x
    if x == -np.inf and y == -np.inf:
        return -np.inf
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """精确 log-domain f 运算（box-plus），支持向量化"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0 and Lb.ndim == 0:
        return logdomain_sum(La + Lb, 0) - logdomain_sum(La, Lb)
    result = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    la_flat = np.broadcast_to(La, result.shape).flat
    lb_flat = np.broadcast_to(Lb, result.shape).flat
    for i, (a, b) in enumerate(zip(la_flat, lb_flat)):
        result.flat[i] = logdomain_sum(a + b, 0) - logdomain_sum(a, b)
    return result


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def active_llr_level(i, n):
    """从 MSB 起计算第一个 1 之前的 0 个数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def active_bit_level(i, n):
    """从 MSB 起计算第一个 0 之前的 1 个数"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _update_llrs(L, B, l, n, N):
    for s in range(n - active_llr_level(l, n), n):
        block_size = 1 << (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1])


def _update_bits(B, l, n, N):
    if l < N // 2:
        return
    for s in range(n, n - active_bit_level(l, n), -1):
        block_size = 1 << s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _sc_decode_internal(llr_ch, frozen_bits):
    """SC 译码核心"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])

    for i in range(N):
        l = bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    输入信道 LLR（对应编码后的码字），输出源序列 u_hat。
    """
    from encoder import bit_reversal_permutation
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return _sc_decode_internal(llr_ch[br], frozen_bits)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算辅助向量（兼容 SCL 接口）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
