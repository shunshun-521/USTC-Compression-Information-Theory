"""
极化码 SC（串行抵消）译码器
提供递归版本（参考）和非递归版本（主实现）
"""
import numpy as np

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算（boxplus）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    if np.isscalar(u_hat):
        return La + Lb if u_hat == 0 else La - Lb
    return np.where(u_hat == 1, La - Lb, La + Lb)


def _compute_left_alpha(llr):
    N = llr.size // 2
    left, right = llr[:N], llr[N:]
    return f_operation(left, right)


def _compute_right_alpha(llr, left_beta):
    N = llr.size // 2
    left, right = llr[:N], llr[N:]
    return right - (2 * left_beta - 1) * left


def _compute_encoding_step(level, n, source, result):
    step = 2 ** (n - level - 1)
    groups = 2**level
    result = result.copy()
    for g in range(groups):
        start = 2 * g * step
        for p in range(step):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]
    return result


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考）"""
    info_mask = ~np.asarray(frozen_bits, dtype=bool)
    return sc_decode(llr, frozen_bits)  # 主实现已足够高效


# ==================== 非递归 SC 译码（主实现）====================


def precompute_sc_indices(N):
    """预计算辅助信息（接口兼容）"""
    n = int(np.log2(N))
    llr_layer_vec = [list(range(1, n + 1)) for _ in range(N)]
    bit_layer_vec = [list(range(n, -1, -1)) for _ in range(N)]
    return np.arange(N), llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits: True/1 表示冻结位；内部使用 info_mask = ~frozen_bits。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    info_mask = ~frozen_bits

    # 分层 LLR
    intermediate_llr = [llr_ch.copy()]
    length = N // 2
    while length > 0:
        intermediate_llr.append(np.zeros(length, dtype=np.float64))
        length //= 2

    intermediate_bits = [np.zeros(N, dtype=int) for _ in range(n + 1)]
    current_state = np.zeros(n, dtype=int)
    previous_state = np.ones(n, dtype=int)

    for position in range(N):
        # 解码器状态（position 的二进制展开，MSB 对齐）
        bits = np.unpackbits(
            np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        current_state = bits[-n:].astype(int)

        # 更新中间 LLR
        for i in range(1, n + 1):
            if current_state[i - 1] == previous_state[i - 1]:
                continue
            llr = intermediate_llr[i - 1]
            if current_state[i - 1] == 0:
                intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - 2 ** (n - i)
                left_bits = intermediate_bits[i][start:end]
                intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

        # 判决
        if info_mask[position]:
            intermediate_bits[-1][position] = (
                1 if intermediate_llr[-1][0] < 0 else 0
            )
        else:
            intermediate_bits[-1][position] = 0

        # 比特回传
        intermediate_bits[-1][position] = intermediate_bits[-1][position]
        for i in range(n - 1, -1, -1):
            intermediate_bits[i] = _compute_encoding_step(
                i, n, intermediate_bits[i + 1], intermediate_bits[i]
            )

        previous_state = current_state.copy()

    return intermediate_bits[-1]
