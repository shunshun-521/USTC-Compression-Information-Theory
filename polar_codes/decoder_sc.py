"""
极化码 SC（串行抵消）译码器
基于逐位状态机的非递归实现（参考 Fast-SSC / SC 文献）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    u_hat = np.asarray(u_hat)
    return Lb - (2.0 * u_hat - 1.0) * La


def _compute_left_alpha(llr):
    half = len(llr) // 2
    return f_operation(llr[:half], llr[half:])


def _compute_right_alpha(llr, left_bits):
    half = len(llr) // 2
    left = llr[:half]
    right = llr[half:]
    return right - (2.0 * left_bits - 1.0) * left


def _compute_encoding_step(level, n, source, result):
    """极化编码单步（用于比特回传）"""
    step = 1 << (n - level - 1)
    groups = 1 << level
    result = result.copy()
    for g in range(groups):
        start = 2 * g * step
        for p in range(step):
            result[p + start] = source[p + start] ^ source[p + start + step]
            result[p + start + step] = source[p + start + step]
    return result


def _position_bits(position, n):
    bits = np.unpackbits(
        np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
    )
    return bits[-n:].astype(np.int8)


def precompute_sc_indices(N):
    """接口兼容：返回占位预计算结构"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    layers = list(range(n))
    return lambda_offset, [layers] * N, [layers] * N


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码。
    frozen_bits[i]=True/1 表示冻结位（强制为 0）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    intermediate_llr = [llr_ch.copy()]
    length = N // 2
    while length > 0:
        intermediate_llr.append(np.zeros(length, dtype=np.float64))
        length //= 2

    intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
    current_state = np.zeros(n, dtype=np.int8)
    previous_state = np.ones(n, dtype=np.int8)

    for position in range(N):
        current_state = _position_bits(position, n)

        for i in range(1, n + 1):
            if current_state[i - 1] == previous_state[i - 1]:
                continue
            llr = intermediate_llr[i - 1]
            if current_state[i - 1] == 0:
                intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (1 << (n - i))
                left_bits = intermediate_bits[i][start:end]
                intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

        if frozen_bits[position]:
            decision = 0
        else:
            decision = 1 if intermediate_llr[-1][0] < 0 else 0

        intermediate_bits[-1][position] = decision
        for i in range(n - 1, -1, -1):
            intermediate_bits[i] = _compute_encoding_step(
                i, n, intermediate_bits[i + 1], intermediate_bits[i]
            )

        previous_state = current_state.copy()

    return intermediate_bits[-1].astype(np.int8)


def sc_decode_recursive(llr, frozen_bits):
    """递归版本（调用非递归实现）"""
    return sc_decode(llr, frozen_bits)


def path_metric_update(pm, llr, u):
    hard = 0 if llr >= 0 else 1
    if u != hard:
        pm += abs(llr)
    return pm
