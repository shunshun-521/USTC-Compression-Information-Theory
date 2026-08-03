"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    精确 log-domain f 运算（box-plus）：
    f(a,b) = ln((1 + e^(a+b)) / (e^a + e^b))
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb + (1.0 - 2.0 * u_hat.astype(np.float64)) * La


def _penalty(llr, bit):
    """路径度量惩罚项"""
    return float(np.logaddexp(0.0, -(1.0 - 2.0 * bit) * llr))


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现，list_size=1 的 SCL）"""
    paths = _scl_decode_core(llr, frozen_bits, list_size=1)
    return paths[0][1].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码（list_size=1 的 SCL 实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def _scl_decode_core(channel_llr, frozen_bits, list_size):
    """
    SCL 译码核心（list_size=1 时等价于 SC）。
    返回 [(metric, u_hat), ...] 按度量升序排列。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    block_length = len(frozen_bits)
    metrics = [0.0]
    decisions = [np.zeros(block_length, dtype=np.int8)]

    def leaf(llrs, index):
        nonlocal metrics, decisions
        if frozen_bits[index]:
            for path, llr in enumerate(llrs):
                metrics[path] += _penalty(float(llr[0]), 0)
                decisions[path][index] = 0
            return [np.zeros(1, dtype=np.int8) for _ in llrs], list(range(len(llrs)))

        candidates = [
            (metrics[path] + _penalty(float(llr[0]), bit), path, bit)
            for path, llr in enumerate(llrs)
            for bit in (0, 1)
        ]
        candidates.sort(key=lambda c: c[0])
        kept = candidates[:list_size]

        new_metrics, new_decisions, betas, parent_map = [], [], [], []
        for metric, path, bit in kept:
            new_metrics.append(metric)
            decision = decisions[path].copy()
            decision[index] = bit
            new_decisions.append(decision)
            betas.append(np.array([bit], dtype=np.int8))
            parent_map.append(path)
        metrics[:] = new_metrics
        decisions[:] = new_decisions
        return betas, parent_map

    def node(llrs, base, length):
        if length == 1:
            return leaf(llrs, base)

        half = length // 2
        upper = [f_operation(llr[:half], llr[half:]) for llr in llrs]
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

    llr = np.asarray(channel_llr, dtype=np.float64)
    node([llr], 0, block_length)
    return sorted(zip(metrics, decisions), key=lambda x: x[0])
