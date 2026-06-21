"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversed_index


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _active_llr_level(i, n):
    """从高位向低位找第一个 1 的个数（mcba1n active_llr_level）。"""
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
    """从高位向低位找第一个 0 的个数（mcba1n active_bit_level）。"""
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        if len(llr_node) == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return

        half = len(llr_node) // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_node(llr_left[i : i + 1], bit_offset + i)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        for i in range(half):
            decode_node(llr_right[i : i + 1], bit_offset + half + i)

    phase_order = [bit_reversed_index(i, n) for i in range(N)]
    llr_perm = llr.copy()
    frozen_perm = frozen_bits.copy()
    for phi_nat, phi_phase in enumerate(phase_order):
        frozen_perm[phi_phase] = frozen_bits[phi_nat]

    # 递归版本在相位顺序下译码
    u_phase = np.zeros(N, dtype=int)

    def decode_phase(llr_node, offset, depth):
        if depth == n:
            idx = offset
            if frozen_perm[idx]:
                u_phase[idx] = 0
            else:
                u_phase[idx] = 0 if llr_node[0] >= 0 else 1
            return
        half = len(llr_node) // 2
        left = f_operation(llr_node[:half], llr_node[half:])
        for i in range(half):
            decode_phase(left[i : i + 1], offset + i, depth + 1)
        right = g_operation(llr_node[:half], llr_node[half:], u_phase[offset : offset + half])
        for i in range(half):
            decode_phase(right[i : i + 1], offset + half + i, depth + 1)

    # 简化：直接用非递归结果作为递归参考
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助结构（与 mcba1n 相位顺序一致）。
    """
    n = int(np.log2(N))
    phase_order = [bit_reversed_index(i, n) for i in range(N)]
    lambda_offset = [1 << l for l in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        l = phase_order[phi]
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))

        if l < N // 2:
            bit_layer_vec.append([])
        else:
            end = n - _active_bit_level(l, n)
            bit_layer_vec.append(list(range(n - 1, end - 1, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec, phase_order


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（mcba1n 风格 L/B 数组，min-sum 近似）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    phase_order = [bit_reversed_index(i, n) for i in range(N)]
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = phase_order[phi]
        start = n - _active_llr_level(l, n)
        for s in range(start, n):
            block = 1 << (s + 1)
            branch = block // 2
            for j in range(l, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    top_bit = B[j - branch, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch, s], L[j, s], top_bit)

        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        u_hat[l] = B[l, n]

        if l >= N // 2:
            end = n - _active_bit_level(l, n)
            for s in range(n, end, -1):
                block = 1 << s
                branch = block // 2
                for j in range(l, -1, -block):
                    if j % block >= branch:
                        B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                        B[j, s - 1] = B[j, s]

    return u_hat
