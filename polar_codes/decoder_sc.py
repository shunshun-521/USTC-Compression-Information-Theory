"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
import math


def f_operation(La, Lb):
    """精确 log-domain f（check-node）运算。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：g(a,b,u) = b + (1-2u)a"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def _penalty(llr, bit):
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 SCL L=1 等价的树遍历）。"""
    paths = _scl_decode_core(llr, frozen_bits, list_size=1)
    return paths[0][1].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归接口：调用 L=1 的 SCL 核心。"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def _scl_decode_core(channel_llr, frozen_bits, list_size):
    """SCL/SC 共享核心（list_size=1 即为 SC）。"""
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    llr = np.asarray(channel_llr, dtype=np.float64)
    N = llr.size
    metrics = [0.0]
    decisions = [np.zeros(N, dtype=np.uint8)]

    def leaf(llrs, index):
        nonlocal metrics, decisions
        if frozen_bits[index]:
            for path, l in enumerate(llrs):
                metrics[path] += _penalty(float(l[0]), 0)
                decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (metrics[path] + _penalty(float(l[0]), bit), path, bit)
            for path, l in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[:list_size]
        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            dec = decisions[path].copy()
            dec[index] = bit
            new_decisions.append(dec)
            betas.append(np.array([bit], dtype=np.uint8))
            parent_map.append(path)
        metrics = new_metrics
        decisions = new_decisions
        return betas, parent_map

    def node(llrs, base, length):
        if length == 1:
            return leaf(llrs, base)
        half = length // 2
        upper = [f_operation(l[:half], l[half:]) for l in llrs]
        beta_upper, map_upper = node(upper, base, half)
        a = [llrs[map_upper[p]][:half] for p in range(len(map_upper))]
        b = [llrs[map_upper[p]][half:] for p in range(len(map_upper))]
        lower = [g_operation(a[p], b[p], beta_upper[p]) for p in range(len(beta_upper))]
        beta_lower, map_lower = node(lower, base + half, half)
        beta_upper = [beta_upper[map_lower[p]] for p in range(len(map_lower))]
        betas = [
            np.concatenate([beta_upper[p] ^ beta_lower[p], beta_lower[p]])
            for p in range(len(beta_lower))
        ]
        parent_map = [map_upper[map_lower[p]] for p in range(len(map_lower))]
        return betas, parent_map

    codewords, _ = node([llr], 0, N)
    return sorted(
        zip(metrics, decisions, codewords, strict=True), key=lambda t: t[0]
    )


def precompute_sc_indices(N):
    """保留给旧接口；SCL 现使用树形核心。"""
    n = int(math.log2(N))
    return [1 << i for i in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]
