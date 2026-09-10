"""
极化码 SC（串行抵消）译码器
递归 SC（参考）与非递归等价的递归实现（高效）
"""
import math
import numpy as np


def prepare_channel_llr(llr_ch):
    """信道 LLR 直接输入，无需置换。"""
    return np.asarray(llr_ch, dtype=np.float64)


# ==================== 基本运算 ====================

def f_operation(La, Lb, llr_max=30.0):
    """box-plus f 运算（向量化）。"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return _cn_op(La, Lb, llr_max)


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _cn_op(x, y, llr_max=30.0):
    x = np.clip(x, -llr_max, llr_max)
    y = np.clip(y, -llr_max, llr_max)
    return np.log(1.0 + np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _vn_op(x, y, u_hat):
    return (1.0 - 2.0 * u_hat) * x + y


def _hard_bit(llr):
    if llr >= 0:
        return 0.0
    return 1.0


# ==================== 递归 SC 译码 ====================

def _polar_decode_sc_recursive(llr_ch, frozen_ind):
    """递归 SC 核心（含 stage 级部分和 u_hat_up）。"""
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]

        x_llr1 = _cn_op(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_recursive(x_llr1, f1)
        x_llr2 = _vn_op(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_recursive(x_llr2, f2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up = (u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)).astype(np.float64)
        u_hat_up = np.concatenate([u_hat1_up, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_ind[0] == 1:
        u_hat = np.array([0.0])
    else:
        u_hat = np.array([_hard_bit(llr_ch[0])])
        if llr_ch[0] == 0:
            u_hat = np.array([1.0])
    return u_hat, u_hat


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    frozen_ind = np.asarray(frozen_bits, dtype=np.float64)
    llr = prepare_channel_llr(llr)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind)
    return u_hat.astype(np.int8)


# ==================== 非递归接口（委托递归实现）====================

def precompute_sc_indices(N):
    """保留接口：返回译码相位顺序。"""
    n = int(math.log2(N))
    order = []
    for i in range(N):
        rev = int("".join(reversed(format(i, f"0{n}b"))), 2)
        order.append(rev)
    return order, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """SC 译码主函数。"""
    return sc_decode_recursive(llr_ch, frozen_bits)
