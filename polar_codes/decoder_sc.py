"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
from encoder import bit_reversal_permutation
from sc_core import sc_tree_decode
import polar_tree_functions as _fn


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    if La.ndim == 0:
        return _fn.f_hf(float(La), float(Lb))
    return np.vectorize(_fn.f_hf)(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    if La.ndim == 0:
        return _fn.g(float(La), float(Lb), int(u_hat))
    return (1.0 - 2.0 * u_hat) * La + Lb


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归接口别名。"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：返回比特倒序译码顺序。"""
    n = int(np.log2(N))
    decode_order = [bit_reversal_permutation(N)[i] for i in range(N)]
    layers = list(range(n))
    return decode_order, [layers] * N, [layers] * N


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主函数。
    信道 LLR 为自然顺序；内部映射到比特倒序因子树。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return sc_tree_decode(llr_ch[br], frozen_bits)


def sc_decode_bit_reversed(llr_ch, frozen_bits):
    """兼容接口。"""
    return sc_decode(llr_ch, frozen_bits)
