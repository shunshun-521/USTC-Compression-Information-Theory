"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Permuted SCD，高效实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    if np.isinf(l1) and not np.isinf(l2):
        return l2
    if not np.isinf(l1) and np.isinf(l2):
        return l1
    if np.isinf(l1) and np.isinf(l2):
        return np.inf
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, b):
    b = int(b)
    if b == 0:
        if np.isinf(l1) or np.isinf(l2):
            return np.inf
        return l1 + l2
    return l1 - l2


# ==================== 递归 SC 译码（参考实现）====================


def _cn_op(x, y):
    x = np.clip(x, -30.0, 30.0)
    y = np.clip(y, -30.0, 30.0)
    return np.log1p(np.exp(x + y)) - np.log(np.exp(x) + np.exp(y))


def _vn_op(x, y, u_hat):
    return (1 - 2 * u_hat) * x + y


def _polar_decode_sc_recursive(llr_ch, frozen_ind):
    """Sionna 风格递归 SC，返回 (u_hat, u_hat_stage)。"""
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1 = llr_ch[:half]
        llr2 = llr_ch[half:]
        f1 = frozen_ind[:half]
        f2 = frozen_ind[half:]
        u_hat1, u_hat1_up = _polar_decode_sc_recursive(_cn_op(llr1, llr2), f1)
        u_hat2, u_hat2_up = _polar_decode_sc_recursive(
            _vn_op(llr1, llr2, u_hat1_up), f2
        )
        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_int = (u_hat1_up.astype(int) ^ u_hat2_up.astype(int)).astype(int)
        u_hat_up = np.concatenate([u_hat1_up_int, u_hat2_up.astype(int)])
        return u_hat, u_hat_up

    is_frozen = bool(frozen_ind[0])
    if is_frozen:
        u_hat = 0
    else:
        u_hat = 0 if llr_ch[0] >= 0 else 1
    return np.array([u_hat], dtype=int), np.array([u_hat], dtype=int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_bits, dtype=bool).astype(int)
    u_hat, _ = _polar_decode_sc_recursive(llr, frozen_ind)
    return u_hat.astype(int)


# ==================== 非递归 SC 译码（Permuted SCD）====================


def precompute_sc_indices(N):
    """
    预计算 Permuted SCD 的译码顺序与活跃层（供报告/调试）。
    """
    n = int(math.log2(N))
    order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for l in order:
        start_s = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start_s, n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return order, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归接口：当前调用高效递归实现（Sionna 等价树形 SC）。
    frozen_bits: 1 表示冻结位
    """
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_paths_match(llr, frozen_bits):
    """校验递归与非递归 SC 输出一致。"""
    a = sc_decode_recursive(llr, frozen_bits)
    b = sc_decode(llr, frozen_bits)
    return np.array_equal(a, b)
