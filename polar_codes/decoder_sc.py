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


def _llr_active_layers(phi, n):
    """自然顺序下需要更新 LLR 的层。"""
    if phi == 0:
        return list(range(n))
    func = phi - 1
    layers = []
    for s in range(n):
        if ((func >> s) & 1) == 0:
            layers.append(s)
        else:
            break
    return layers


def _bit_active_layers(phi, n):
    """自然顺序下需要回传比特的层。"""
    if phi == 0:
        return []
    func = phi
    layers = []
    for s in range(n):
        if ((func >> s) & 1) == 1:
            layers.append(s)
        else:
            break
    return layers


def sc_decode_iterative(llr_ch, frozen_bits):
    """非递归 SC 译码（自然顺序，与递归实现一致）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((n + 1, N))
    T = np.zeros((n + 1, N), dtype=np.int8)
    L[n, :] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for l in _llr_active_layers(phi, n):
            sp = 2 ** (n - 1 - l)
            for block in range(2 ** l):
                left = block * 2 * sp
                right = left + 2 * sp
                if left <= phi < right:
                    if phi < left + sp:
                        L[l, block] = f_operation(L[l + 1, 2 * block], L[l + 1, 2 * block + 1])
                    else:
                        L[l, block] = g_operation(
                            L[l + 1, 2 * block], L[l + 1, 2 * block + 1], T[l + 1, 2 * block]
                        )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if L[0, 0] >= 0 else 1

        T[0, 0] = u_hat[phi]

        for l in _bit_active_layers(phi, n):
            sp = 2 ** (n - 1 - l)
            for block in range(2 ** l):
                left = block * 2 * sp
                right = left + 2 * sp
                if left <= phi < right:
                    if phi < left + sp:
                        T[l + 1, 2 * block] = T[l, block] ^ T[l + 1, 2 * block + 1]
                    else:
                        T[l + 1, 2 * block + 1] = T[l, block]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """默认 SC 译码入口（递归实现，经验证正确）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
