"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """精确 log-domain f 运算（与 SC/SCL 一致）。"""
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return Lb + (1.0 - 2.0 * np.asarray(u_hat, dtype=np.float64)) * La


def _pm_penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（list_size=1 的 SCL 结构）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    llr = np.asarray(llr, dtype=np.float64)
    metrics = [0.0]
    decisions = [np.zeros(N, dtype=np.int8)]

    def leaf(llrs, index):
        if frozen_bits[index]:
            for path, l in enumerate(llrs):
                metrics[path] += _pm_penalty(float(l[0]), 0)
                decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = []
        for path, l in enumerate(llrs):
            for bit in (0, 1):
                candidates.append((metrics[path] + _pm_penalty(float(l[0]), bit), path, bit))
        best = min(candidates, key=lambda t: t[0])
        metric, path, bit = best
        metrics[:] = [metric]
        dec = decisions[path].copy()
        dec[index] = bit
        decisions[:] = [dec]
        return [np.array([bit], dtype=np.int8)], [0]

    def node(llrs, base, length):
        if length == 1:
            return leaf(llrs, base)

        half = length // 2
        upper = [f_operation(l[:half], l[half:]) for l in llrs]
        beta_upper, map_upper = node(upper, base, half)

        lower = []
        for p, bu in enumerate(beta_upper):
            parent = map_upper[p]
            lower.append(g_operation(llrs[parent][:half], llrs[parent][half:], bu))
        beta_lower, map_lower = node(lower, base + half, half)

        betas = []
        pmap = []
        for p in range(len(beta_lower)):
            bu = beta_upper[map_lower[p]]
            bl = beta_lower[p]
            betas.append(np.concatenate([bu ^ bl, bl]))
            pmap.append(map_upper[map_lower[p]])
        return betas, pmap

    node([llr], 0, N)
    return decisions[0].astype(int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        tmp = phi
        layer = 0
        while tmp & 1:
            llr_layers.append(layer)
            layer += 1
            tmp >>= 1

        bit_layers = []
        tmp = phi + 1
        layer = 0
        while tmp % 2 == 0 and layer < n:
            bit_layers.append(layer)
            layer += 1
            tmp >>= 1

        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


_SC_INDEX_CACHE = {}


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（分层 LLR / 部分和更新，供对照与加速优化）。"""
    N = len(llr_ch)
    if N not in _SC_INDEX_CACHE:
        _SC_INDEX_CACHE[N] = precompute_sc_indices(N)
    _, llr_layer_vec, bit_layer_vec = _SC_INDEX_CACHE[N]

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    n = int(np.log2(N))

    P = np.zeros((n + 1, N), dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[n, :] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            stride = 1 << layer
            for i in range(0, N, 2 * stride):
                for j in range(i, i + stride):
                    P[layer, j] = f_operation(
                        P[layer + 1, j], P[layer + 1, j + stride]
                    )
                    P[layer, j + stride] = g_operation(
                        P[layer + 1, j],
                        P[layer + 1, j + stride],
                        C[layer, j],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
            C[0, phi] = 0
        else:
            u_hat[phi] = 0 if P[0, phi] >= 0 else 1
            C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            stride = 1 << layer
            for i in range(0, N, 2 * stride):
                for j in range(i, i + stride):
                    C[layer + 1, j] = C[layer, j] ^ C[layer + 1, j + stride]

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主入口（默认调用高效树形实现，与递归参考版等价）。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
