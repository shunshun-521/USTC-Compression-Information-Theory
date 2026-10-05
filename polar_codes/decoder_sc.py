"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）与基于因子图遍历的高效 SC 译码（主实现）
"""
import math
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似 f 运算（SCL/BP 等模块复用）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return (1 - 2 * u_hat) * La + Lb


def _f_hf(L1, L2):
    s1 = np.sign(L1) if L1 != 0 else 1.0
    s2 = np.sign(L2) if L2 != 0 else 1.0
    return s1 * s2 * min(abs(L1), abs(L2))


def _g(L1, L2, u):
    return (1 - 2 * u) * L1 + L2


def _all_filled(x):
    return not np.any(np.isnan(x))


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（对数域 min-sum f）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset):
        n = len(llr_node)
        if n == 1:
            idx = bit_offset
            u_hat[idx] = 0 if frozen_bits[idx] else (0 if llr_node[0] >= 0 else 1)
            return
        half = n // 2
        llr_left = f_operation(llr_node[:half], llr_node[half:])
        decode_node(llr_left, bit_offset)
        u_left = u_hat[bit_offset : bit_offset + half]
        llr_right = g_operation(llr_node[:half], llr_node[half:], u_left)
        decode_node(llr_right, bit_offset + half)

    decode_node(llr, 0)
    return u_hat


# ==================== 非递归 SC 译码（主实现）====================


def precompute_sc_indices(N):
    """兼容接口：返回码长与层数"""
    return list(range(N)), int(math.log2(N))


def _sc_decode_factor_graph(y_llr, info_indices, frozen_value=0):
    """
    基于因子图深度优先的 SC 译码（与 u @ G_N 编码配套，G_N = F^{\\otimes n}）。
    """
    y_llr = np.asarray(y_llr, dtype=np.float64)
    N = y_llr.size
    n = int(math.log2(N))
    info_set = set(int(i) for i in info_indices)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        p0, p1, p2, p3 = position
        span = 2 ** (p2 - p0)
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1][p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2 : p1 + span]

        if _all_filled(up_bit):
            position = _sc_up(position)
            continue

        if _all_filled(right_bit):
            up_bit_new = np.array([(left_bit[i] + right_bit[i]) % 2 for i in range(len(left_bit))])
            up_bit_new = np.vstack([up_bit_new, right_bit]).reshape(1, -1)[0]
            bit_matrix[p0][p1 : p1 + span] = up_bit_new
            continue

        if _all_filled(right_llr):
            if p0 == p2 - 1:
                right_pos = p1 + 1
                if right_pos in info_set:
                    right_bit_val = 0 if right_llr[0] >= 0 else 1
                else:
                    right_bit_val = frozen_value
                bit_matrix[p0 + 1][p1 + span // 2 : p1 + span] = right_bit_val
            else:
                position = [p0 + 1, p1 + 2 ** (p2 - 1 - p0), p2, p3]
            continue

        if _all_filled(left_bit):
            right_llr_new = np.array([_g(up_llr[i], up_llr[i + len(left_bit)], left_bit[i]) for i in range(len(left_bit))])
            llr_matrix[p0 + 1][p1 + span // 2 : p1 + span] = right_llr_new
            continue

        if not _all_filled(left_llr):
            left_llr_new = np.array([_f_hf(up_llr[i], up_llr[i + len(left_llr)]) for i in range(len(left_llr))])
            llr_matrix[p0 + 1][p1 : p1 + span // 2] = left_llr_new
            continue

        if p0 == p2 - 1:
            left_pos = p1
            if left_pos in info_set:
                left_bit_val = 0 if left_llr[0] >= 0 else 1
            else:
                left_bit_val = frozen_value
            bit_matrix[p0 + 1][p1 : p1 + span // 2] = left_bit_val
        else:
            position = [p0 + 1, p1, p2, p3]

    return bit_matrix[n].astype(int)


def _sc_up(position):
    p0, p1, p2, p3 = position
    p0 -= 1
    p1 = int(np.floor(p1 / (2 ** (p2 - p0))) * (2 ** (p2 - p0)))
    return [p0, p1, p2, p3]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits: 1 表示冻结位，0 表示信息位。
    """
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(frozen_bits)
    info_indices = np.where(frozen_bits == 0)[0]
    return _sc_decode_factor_graph(llr_ch, info_indices, frozen_value=0)


def sc_decode_channel(llr_ch, frozen_bits):
    """信道 LLR 接口（与编码输出比特顺序一致）。"""
    return sc_decode(llr_ch, frozen_bits)
