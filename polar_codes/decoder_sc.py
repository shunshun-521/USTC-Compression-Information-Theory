"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return Lb + (1 - 2 * u_hat) * La


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


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


def _update_llrs(L, B, l, n, N):
    for s in range(n - _active_llr_level(l, n), n):
        block = 2 ** (s + 1)
        branch = block // 2
        for j in range(l, N, block):
            if j % block < branch:
                L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
            else:
                L[j, s + 1] = g_operation(L[j - branch, s], L[j, s], B[j - branch, s + 1])


def _update_bits(B, l, n, N):
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block = 2 ** s
        branch = block // 2
        for j in range(l, -1, -block):
            if j % block >= branch:
                B[j - branch, s - 1] = int(B[j, s]) ^ int(B[j - branch, s])
                B[j, s - 1] = B[j, s]


def _xor_combine(left, right):
    left = list(left)
    right = list(right)
    res = [(left[i] + right[i]) % 2 for i in range(len(left))]
    res.extend(right)
    return res


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（HETSN 风格参考实现）。"""
    N = len(llr)
    n = int(np.log2(N)) + 1
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])
    node_values = [0] * N

    def decode_node(y, depth, node):
        if depth == n - 1:
            if node in frozen_set:
                node_values[node] = 0
                return [0]
            bit = 1 if y[0] < 0 else 0
            node_values[node] = bit
            return [bit]

        half = len(y) // 2
        l1, l2 = y[:half], y[half:]
        arr1 = decode_node(f_operation(l1, l2), depth + 1, 2 * node)
        arr2 = decode_node(g_operation(l1, l2, np.array(arr1)), depth + 1, 2 * node + 1)
        return _xor_combine(arr1, arr2)

    decode_node(np.asarray(llr, dtype=np.float64), 0, 0)
    return np.array(node_values, dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [2 ** i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        i = 0
        while i < n:
            if (phi >> i) & 1 == 0:
                layers.append(i)
                break
            i += 1
        for j in range(i + 1, n):
            layers.append(j)
        llr_layer_vec.append(layers)

        if phi % 2 == 0:
            b_layers = []
            i = 0
            while True:
                b_layers.append(i)
                if (phi >> (i + 1)) & 1 == 0:
                    break
                i += 1
            bit_layer_vec.append(b_layers)
        else:
            b_layers = []
            i = 0
            while True:
                b_layers.append(i)
                if (phi >> i) & 1 == 0:
                    break
                i += 1
            bit_layer_vec.append(b_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（高效迭代实现）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1))
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, N)
        if frozen_bits[l]:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(int)
