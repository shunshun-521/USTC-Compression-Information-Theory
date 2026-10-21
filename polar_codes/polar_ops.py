"""极化码 f/g 核运算"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def f_boxplus(a, b):
    """精确 log-domain f"""
    return np.asarray(np.logaddexp(0.0, a + b) - np.logaddexp(a, b), dtype=np.float64)
