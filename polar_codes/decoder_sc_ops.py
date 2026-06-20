"""SC 译码基本运算"""
import numpy as np


def f_operation(La, Lb):
  """
  min-sum 近似的 f 运算：
  f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  """
  return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
  """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
  return (1 - 2 * u_hat) * La + Lb
