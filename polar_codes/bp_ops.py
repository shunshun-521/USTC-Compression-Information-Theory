"""对 decoder_sc.f_operation 的 min-sum 近似，供 BP 译码使用。"""
import numpy as np


def f_min_sum(La, Lb, alpha=1.0):
    """min-sum 近似 f 运算。"""
    return alpha * np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))
