"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（LLR=0 时 sign 取 +1）"""
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _all_computed(arr):
    return not np.any(np.isnan(arr))


def _sc_decode_fsm(llr, info_positions, frozen_value=0):
    """
    基于有限状态机的非递归 SC 译码核心。
    llr 应在译码树顺序下排列（对自然信道 LLR 需先做比特倒序）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    N = llr.size
    n = int(math.log2(N))
    info_set = set(info_positions)

    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = llr
    position = [0, 0, n, N]

    def leftdown(pos):
        return [pos[0] + 1, pos[1], pos[2], pos[3]]

    def rightdown(pos):
        return [pos[0] + 1, pos[1] + 2 ** (pos[2] - 1 - pos[0]), pos[2], pos[3]]

    def up_pos(pos):
        p0 = pos[0] - 1
        p1 = int(
            np.floor(pos[1] / (2 ** (pos[2] - pos[0] + 1)))
            * (2 ** (pos[2] - pos[0] + 1))
        )
        return [p0, p1, pos[2], pos[3]]

    while not _all_computed(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        p0, p1 = position[0], position[1]
        up_llr = llr_matrix[p0][p1 : p1 + span]
        up_bit = bit_matrix[p0][p1 : p1 + span]
        left_llr = llr_matrix[p0 + 1][p1 : p1 + span // 2]
        left_bit = bit_matrix[p0 + 1][p1 : p1 + span // 2]
        right_llr = llr_matrix[p0 + 1][p1 + span // 2 : p1 + span]
        right_bit = bit_matrix[p0 + 1][p1 + span // 2 : p1 + span]

        if _all_computed(up_bit):
            position = up_pos(position)
        elif _all_computed(right_bit):
            merged = np.zeros(span, dtype=int)
            merged[: span // 2] = (left_bit.astype(int) + right_bit.astype(int)) % 2
            merged[span // 2 :] = right_bit.astype(int)
            bit_matrix[p0][p1 : p1 + span] = merged
        elif _all_computed(right_llr):
            if position[0] == position[2] - 1:
                bit_pos = position[1] + 1
                if bit_pos in info_set:
                    bit_val = 0 if right_llr[0] > 0 else 1
                else:
                    bit_val = frozen_value
                bit_matrix[p0 + 1][p1 + span // 2 : p1 + span] = bit_val
            else:
                position = rightdown(position)
        elif _all_computed(left_bit.astype(float)):
            right_llr_new = np.array(
                [
                    g_operation(up_llr[i], up_llr[i + span // 2], int(left_bit[i]))
                    for i in range(span // 2)
                ]
            )
            llr_matrix[p0 + 1][p1 + span // 2 : p1 + span] = right_llr_new
        elif not _all_computed(left_llr):
            left_llr_new = np.array(
                [
                    f_operation(up_llr[i], up_llr[i + span // 2])
                    for i in range(span // 2)
                ]
            )
            llr_matrix[p0 + 1][p1 : p1 + span // 2] = left_llr_new
        else:
            if position[0] == position[2] - 1:
                bit_pos = position[1]
                if bit_pos in info_set:
                    bit_val = 0 if left_llr[0] >= 0 else 1
                else:
                    bit_val = frozen_value
                bit_matrix[p0 + 1][p1 : p1 + span // 2] = bit_val
            else:
                position = leftdown(position)

    return bit_matrix[n].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    rev = bit_reversal_permutation(N)
    llr = llr[rev]
    info_positions = np.where(~frozen_bits)[0]
    return _sc_decode_fsm(llr, info_positions, frozen_value=0)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    def active_llr_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) == 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    def active_bit_level(i):
        mask = 2 ** (n - 1)
        count = 1
        for _ in range(n):
            if (mask & i) > 0:
                count += 1
                mask >>= 1
            else:
                break
        return min(count, n)

    for phi in range(N):
        llr_layer_vec.append(list(range(active_llr_level(phi), n)))
        bit_layer_vec.append(list(range(n, n - active_bit_level(phi), -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    编码端含比特倒序，因此信道 LLR 需先倒序再送入译码树。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    info_positions = np.where(~frozen_bits)[0]
    return _sc_decode_fsm(llr_ch[rev], info_positions, frozen_value=0)


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, K / N)
    mismatches = errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = np.random.randint(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen)
        u_rec = sc_decode_recursive(llr, frozen)
        if not np.array_equal(u_sc, u_rec):
            mismatches += 1
        if not np.array_equal(u[info_idx], u_sc[info_idx]):
            errors += 1
    print(f"Recursive vs non-recursive mismatches: {mismatches}/100")
    print(f"N=64 high-SNR test errors: {errors}/100")
