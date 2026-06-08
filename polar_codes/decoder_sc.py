"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math

import numpy as np

CHECK_NODE_TANH_THRES = 44


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（向量化）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _llr_check_node(llr_1, llr_2):
    """min-sum 近似的 f 运算（数值稳定）"""
    return float(f_operation(llr_1, llr_2))


def _calc_llr(i, n, llr, u_est):
    """
    递归计算第 i 个极化信道的 LLR（参考 RQC 慢速 SC 实现）。
    llr: 当前子问题的信道 LLR 向量（长度 2^k）
    u_est: 已译码的前 i 个比特
    """
    N = len(llr)
    if i == 0 and N == 1:
        return float(llr[0])

    if i % 2 == 0:
        llr_1 = _calc_llr(
            i // 2,
            n - 1,
            llr[: N // 2],
            (u_est[::2] ^ u_est[1::2])[: i // 2],
        )
        llr_2 = _calc_llr(
            i // 2,
            n - 1,
            llr[N // 2 :],
            u_est[1::2][: i // 2],
        )
        return _llr_check_node(llr_1, llr_2)

    prev = int(u_est[-1])
    llr_1 = _calc_llr(
        (i - 1) // 2,
        n - 1,
        llr[: N // 2],
        (u_est[:-1:2] ^ u_est[1:-1:2])[: (i - 1) // 2],
    )
    llr_2 = _calc_llr(
        (i - 1) // 2,
        n - 1,
        llr[N // 2 :],
        u_est[1::2][: (i - 1) // 2],
    )
    return llr_2 + ((-1) ** prev) * llr_1


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现，O(N^2) LLR 计算）
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
        else:
            llr_i = _calc_llr(i, n, llr, u_hat[:i])
            u_hat[i] = 0 if llr_i >= 0 else 1

    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（与 Arikan 快速 SC 层索引一致）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        bits = format(phi, f"0{n}b")
        for layer in range(n):
            if bits[n - 1 - layer] == "0":
                layers.append(layer)
        llr_layer_vec.append(layers)

        if phi % 2 == 0:
            bit_layers = list(range(n))
        else:
            bit_layers = []
            for layer in range(n):
                if bits[n - 1 - layer] == "1":
                    bit_layers.append(layer)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（快速 LLR 缓存实现）
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)
    llr_array = np.zeros(N * (n + 1), dtype=np.float64)
    is_calc = np.zeros(N * (n + 1), dtype=bool)

    def get_problem_i(idx):
        slice_idx = idx // N
        modulus = 2 ** (n - slice_idx)
        return idx % modulus

    def get_descendants(idx):
        slice_idx = idx // N
        slice_i = idx - slice_idx * N
        sub_len = 2 ** (n - slice_idx)
        sub_start = (slice_i // sub_len) * sub_len
        sub_i = idx % sub_len
        left = (slice_idx + 1) * N + sub_start + (sub_i // 2)
        right = left + 2 ** (n - slice_idx - 1)
        return left, right

    def fast_llr(idx, y, u_est):
        if is_calc[idx]:
            return llr_array[idx]

        problem_i = get_problem_i(idx)
        cur_n = len(y)

        if problem_i == 0 and cur_n == 1:
            llr_array[idx] = float(y[0])
        elif problem_i % 2 == 0:
            left_desc, right_desc = get_descendants(idx)
            half = cur_n // 2
            llr_1 = fast_llr(
                left_desc,
                y[:half],
                (u_est[::2] ^ u_est[1::2])[: problem_i // 2],
            )
            llr_2 = fast_llr(
                right_desc,
                y[half:],
                u_est[1::2][: problem_i // 2],
            )
            llr_array[idx] = _llr_check_node(llr_1, llr_2)
        else:
            left_desc, right_desc = get_descendants(idx)
            half = cur_n // 2
            llr_1 = fast_llr(
                left_desc,
                y[:half],
                (u_est[:-1:2] ^ u_est[1:-1:2])[: (problem_i - 1) // 2],
            )
            llr_2 = fast_llr(
                right_desc,
                y[half:],
                u_est[1::2][: (problem_i - 1) // 2],
            )
            llr_array[idx] = llr_2 + ((-1) ** int(u_est[-1])) * llr_1

        is_calc[idx] = True
        return llr_array[idx]

    for i in range(N):
        if frozen_bits[i]:
            u_hat[i] = 0
        else:
            llr_i = fast_llr(i, llr_ch, u_hat[:i])
            u_hat[i] = 0 if llr_i >= 0 else 1

    return u_hat
