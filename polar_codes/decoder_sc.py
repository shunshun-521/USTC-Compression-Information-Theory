"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

# ==================== 基本运算 ====================


def _bit_reversed(x, n):
    """比特倒序索引。"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层。"""
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
    """比特回传起始层。"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _logdomain_diff(x, y):
    if x > y:
        return x + np.log1p(-np.exp(y - x))
    return y + np.log1p(-np.exp(x - y))


def _upper_llr(l1, l2):
    return _logdomain_diff(l1 + l2, 0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    if b == 0:
        return l1 + l2
    return l1 - l2


def f_operation(La, Lb):
    """f 运算（box-plus，支持标量与数组）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        la, lb = float(La), float(Lb)
        la = np.clip(la, -30, 30)
        lb = np.clip(lb, -30, 30)
        return np.log1p(np.exp(la + lb)) - np.log(np.exp(la) + np.exp(lb))
    la = np.clip(La, -30, 30)
    lb = np.clip(Lb, -30, 30)
    return np.log1p(np.exp(la + lb)) - np.log(np.exp(la) + np.exp(lb))


def f_operation_minsum(La, Lb):
    sign = 1.0 if La >= 0 else -1.0
    sign *= 1.0 if Lb >= 0 else -1.0
    return sign * min(abs(La), abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（支持标量与数组）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
            else:
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = g_operation(
                    L[j, s], L[j - branch_size, s], top_bit
                )


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


# ==================== 非递归 SC 译码 ====================


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（按比特倒序逐位处理）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch

    for i in range(N):
        l = _bit_reversed(i, n)
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 XOR 编码器配对）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_node(llr_node, frozen_node):
        n = len(llr_node)
        if n == 1:
            if frozen_node[0]:
                u = 0
            else:
                u = 0 if llr_node[0] >= 0 else 1
            return np.array([u], dtype=int), np.array([u], dtype=float)

        half = n // 2
        llr1 = llr_node[:half]
        llr2 = llr_node[half:]
        fr1 = frozen_node[:half]
        fr2 = frozen_node[half:]

        llr_u = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = decode_node(llr_u, fr1)

        llr_v = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = decode_node(llr_v, fr2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_up_left = (u_hat1_up.astype(int) ^ u_hat2_up.astype(int)).astype(float)
        u_hat_up = np.concatenate([u_up_left, u_hat2_up])
        return u_hat, u_hat_up

    u_hat, _ = decode_node(llr, frozen_bits)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（接口兼容）。"""
    n = int(np.log2(N))
    lambda_offset = [2 ** i - 1 for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [[] for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主入口（默认递归，与 XOR 编码器配对）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
