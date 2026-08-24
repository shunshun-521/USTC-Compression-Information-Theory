"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


CHECK_NODE_TANH_THRES = 44


# ==================== 基本运算 ====================

def _sign_llr(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """min-sum 近似的 f 运算。"""
    return _sign_llr(La) * _sign_llr(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算。"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _llr_check_node_operation(llr_1, llr_2):
    """f 运算：统一使用 min-sum 近似（数值稳定）。"""
    return float(f_operation(llr_1, llr_2))


def _get_problem_i(i, n, N):
    slice_idx = i // N
    modulus = 1 << (n - slice_idx)
    return i % modulus


def _get_descendants(i, n, N):
    slice_idx = i // N
    slice_i = i - slice_idx * N
    subproblem_len = 1 << (n - slice_idx)
    subproblem_start = (slice_i // subproblem_len) * subproblem_len
    subproblem_i = i % subproblem_len
    left_desc = (slice_idx + 1) * N + subproblem_start + (subproblem_i // 2)
    right_desc = left_desc + (1 << (n - slice_idx - 1))
    return left_desc, right_desc


def _slow_llr(i, llr_work, u_est, n, code_N):
    """递归 SC LLR（浮点信道 LLR 切片）。"""
    N = len(llr_work)
    if i == 0 and N == 1:
        return float(llr_work[0])

    half = N // 2
    if i % 2 == 0:
        llr_1 = _slow_llr(
            i // 2, llr_work[:half],
            (u_est[::2] ^ u_est[1::2])[:i // 2], n, code_N,
        )
        llr_2 = _slow_llr(
            i // 2, llr_work[half:],
            u_est[1::2][:i // 2], n, code_N,
        )
        return _llr_check_node_operation(llr_1, llr_2)

    llr_1 = _slow_llr(
        (i - 1) // 2, llr_work[:half],
        (u_est[:-1:2] ^ u_est[1:-1:2])[:(i - 1) // 2], n, code_N,
    )
    llr_2 = _slow_llr(
        (i - 1) // 2, llr_work[half:],
        u_est[1::2][:(i - 1) // 2], n, code_N,
    )
    return llr_2 + ((-1) ** u_est[-1]) * llr_1


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)
    frozen_positions = set(np.where(frozen_bits == 1)[0])

    for i in range(N):
        if i in frozen_positions:
            u_hat[i] = 0
        else:
            llr_i = _slow_llr(i, llr, u_hat[:i], n, N)
            u_hat[i] = 0 if llr_i > 0 else 1

    return u_hat


def _fast_llr(i, llr_work, u_est, llr_array, is_calc, n, N):
    """快速 LLR 计算：llr_work 为当前子问题的信道 LLR 切片（与 RQC 的 y 切片同构）。"""
    if is_calc[i]:
        return llr_array[i]

    problem_i = _get_problem_i(i, n, N)
    sub_len = len(llr_work)

    if problem_i == 0 and sub_len == 1:
        llr_array[i] = float(llr_work[0])
    else:
        half = sub_len // 2
        left_desc, right_desc = _get_descendants(i, n, N)

        if problem_i % 2 == 0:
            llr_1 = _fast_llr(
                left_desc, llr_work[:half],
                (u_est[::2] ^ u_est[1::2])[:problem_i // 2],
                llr_array, is_calc, n, N,
            )
            llr_2 = _fast_llr(
                right_desc, llr_work[half:],
                u_est[1::2][:problem_i // 2],
                llr_array, is_calc, n, N,
            )
            llr_array[i] = _llr_check_node_operation(llr_1, llr_2)
        else:
            llr_1 = _fast_llr(
                left_desc, llr_work[:half],
                (u_est[:-1:2] ^ u_est[1:-1:2])[:problem_i // 2],
                llr_array, is_calc, n, N,
            )
            llr_2 = _fast_llr(
                right_desc, llr_work[half:],
                u_est[1::2][:problem_i // 2],
                llr_array, is_calc, n, N,
            )
            llr_array[i] = llr_2 + ((-1) ** u_est[-1]) * llr_1

    is_calc[i] = True
    return llr_array[i]


def precompute_sc_indices(N):
    """预计算辅助向量（接口兼容）。"""
    n = int(math.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        tmp = phi
        layer = 0
        while tmp % 2 == 1:
            layers_llr.append(layer)
            tmp >>= 1
            layer += 1
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        if phi % 2 == 1:
            tmp = phi
            layer = 0
            while tmp % 2 == 1:
                layers_bit.append(layer)
                tmp >>= 1
                layer += 1
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    frozen_positions = set(np.where(frozen_bits == 1)[0])
    u_hat = np.full(N, -1, dtype=int)

    is_calc = [False] * (N * (n + 1))
    llr_array = np.zeros(N * (n + 1), dtype=np.float64)

    for i in range(N):
        if i in frozen_positions:
            u_hat[i] = 0
        else:
            llr_i = _fast_llr(i, llr_ch, u_hat[:i], llr_array, is_calc, n, N)
            u_hat[i] = 0 if llr_i > 0 else 1

    return u_hat.astype(int)
