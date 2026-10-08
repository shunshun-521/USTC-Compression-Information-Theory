"""
极化码 SC（串行抵消）译码器
提供递归 SCL 树（list_size=1）与非递归接口
"""
import math
import numpy as np


def f_operation(La, Lb):
    """精确 log 域 f：ln((1+e^{a+b})/(e^a+e^b))"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算：b + (1-2u)*a"""
    u = np.asarray(u_hat, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u) * La


def _penalty(llr: float, bit: int) -> float:
    return float(np.logaddexp(0.0, -(1.0 - 2 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（与 SCL list_size=1 等价）"""
    paths = _scl_decode_core(llr, frozen_bits, list_size=1)
    return paths[0][1].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归接口：调用高效 SCL 核心（L=1）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：供报告/扩展使用"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        p = phi
        layer = 0
        while p & 1:
            p >>= 1
            layer += 1
        llr_layer_vec.append(
            list(range(n - 1, -1, -1)) if phi == 0 else list(range(n - 1, layer - 1, -1))
        )
        if phi % 2 == 0:
            bit_layer_vec.append(list(range(n - 1, -1, -1)))
        else:
            p2 = phi
            lb = 0
            while p2 & 1:
                p2 >>= 1
                lb += 1
            bit_layer_vec.append(list(range(lb - 1, -1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _scl_decode_core(channel_llr, frozen_bits, list_size):
    """SCL 树形译码核心，list_size=1 即为 SC"""
    frozen = np.asarray(frozen_bits, dtype=bool)
    N = frozen.size
    llr = np.asarray(channel_llr, dtype=np.float64)
    metrics = [0.0]
    decisions = [np.zeros(N, dtype=np.uint8)]

    def leaf(llrs, index):
        nonlocal metrics, decisions
        if frozen[index]:
            for path, l in enumerate(llrs):
                metrics[path] += _penalty(float(l[0]), 0)
                decisions[path][index] = 0
            return [np.zeros(1, dtype=np.uint8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (metrics[path] + _penalty(float(l[0]), bit), path, bit)
            for path, l in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda x: x[0])
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
        beta_u, map_u = node(upper, base, half)
        a = [llrs[map_u[p]][:half] for p in range(len(map_u))]
        b = [llrs[map_u[p]][half:] for p in range(len(map_u))]
        lower = [g_operation(a[p], b[p], beta_u[p]) for p in range(len(beta_u))]
        beta_l, map_l = node(lower, base + half, half)
        beta_u = [beta_u[map_l[p]] for p in range(len(map_l))]
        betas = [
            np.concatenate([beta_u[p] ^ beta_l[p], beta_l[p]]) for p in range(len(beta_l))
        ]
        parent_map = [map_u[map_l[p]] for p in range(len(map_l))]
        return betas, parent_map

    codewords, _ = node([llr], 0, N)
    return sorted(zip(metrics, decisions, codewords), key=lambda x: x[0])
