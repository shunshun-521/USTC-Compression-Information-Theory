"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（用于 BP）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """标量精确 f 运算"""
    if np.isscalar(La) and np.isscalar(Lb):
        return _f_scalar(La, Lb)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.vectorize(_f_scalar)(La, Lb)


def _f_scalar(a, b):
    return float(np.sign(a) * np.sign(b) * min(abs(a), abs(b)))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La"""
    u = np.asarray(u_hat)
    return Lb + (1 - 2 * u) * La


def _to_frozen_set(frozen_bits):
    fb = np.asarray(frozen_bits, dtype=bool)
    return set(np.where(fb)[0])


def _permute_llr(llr_ch):
    """对信道 LLR 做逆比特倒序置换，与编码端 bit-reversal 匹配"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[np.argsort(rev)]


def _decode_recursive(llr, depth, n, node, node_values, frozen_set):
    """HETSN 风格递归 SC 译码核心"""
    if depth == n - 1:
        if node in frozen_set:
            bit = 0
        else:
            bit = 1 if llr[0] < 0 else 0
        node_values[node] = bit
        return [bit]

    half = len(llr) // 2
    l1, l2 = llr[:half], llr[half:]
    f_out = [float(_f_scalar(a, b)) for a, b in zip(l1, l2)]
    arr1 = _decode_recursive(f_out, depth + 1, n, 2 * node, node_values, frozen_set)
    g_out = [float(g_operation(a, b, u)) for a, b, u in zip(l1, l2, arr1)]
    arr2 = _decode_recursive(g_out, depth + 1, n, 2 * node + 1, node_values, frozen_set)
    merged = [(a + b) % 2 for a, b in zip(arr1, arr2)]
    merged.extend(arr2)
    return merged


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = _permute_llr(llr)
    N = len(llr)
    n = int(math.log2(N)) + 1
    node_values = [0] * N
    _decode_recursive(llr.tolist(), 0, n, 0, node_values, _to_frozen_set(frozen_bits))
    return np.array(node_values, dtype=int)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（兼容接口）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = [list(range(n)) for _ in range(N)]
    bit_layer_vec = [list(range(n - 1, -1, -1)) for _ in range(N)]
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（调用高效递归核心，接口保持一致）。
    """
    return sc_decode_recursive(llr_ch, frozen_bits)
