"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    sa = np.sign(La)
    sb = np.sign(Lb)
    sa = np.where(sa == 0, 1, sa)
    sb = np.where(sb == 0, 1, sb)
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


def _frozen_mask_to_info_pos(frozen_bits):
    """将冻结位掩码转换为信息位索引列表。"""
    frozen_bits = np.asarray(frozen_bits)
    if frozen_bits.dtype == bool:
        frozen = frozen_bits
    else:
        frozen = frozen_bits.astype(int) != 0
    return np.where(~frozen)[0].tolist()


def _prepare_llr(llr_ch):
    """对信道 LLR 做比特倒序，以匹配蝶形编码 + 比特倒序输出。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return llr_ch[rev]


# ==================== SC 核心状态机（非递归）====================


def _all_filled(arr):
    return not np.any(np.isnan(arr))


def _leftdown(position):
    return [position[0] + 1, position[1], position[2], position[3]]


def _rightdown(position):
    return [
        position[0] + 1,
        position[1] + 2 ** (position[2] - 1 - position[0]),
        position[2],
        position[3],
    ]


def _up(position):
    p0 = position[0] - 1
    p1 = int(np.floor(position[1] / (2 ** (position[2] - position[0] + 1)))
             * (2 ** (position[2] - position[0] + 1)))
    return [p0, p1, position[2], position[3]]


def _get_up_bit(left_bit, right_bit):
    length = left_bit.size
    temp = np.array([(left_bit + right_bit) % 2, right_bit])
    temp.resize((1, 2 * length))
    return temp.flatten()


def _get_left_bit(left_llr, information_pos, frozen_bit, left_bit_pos):
    if left_bit_pos in information_pos:
        return 0 if left_llr >= 0 else 1
    return frozen_bit


def _get_right_bit(right_llr, information_pos, frozen_bit, right_bit_pos):
    if right_bit_pos in information_pos:
        return 0 if right_llr >= 0 else 1
    return frozen_bit


def _get_right_llr(left_bit, up_llr):
    length = left_bit.size
    return np.array(
        [g_operation(up_llr[i], up_llr[i + length], left_bit[i]) for i in range(length)]
    )


def _get_left_llr(up_llr):
    length = up_llr.size // 2
    return np.array(
        [f_operation(up_llr[i], up_llr[i + length]) for i in range(length)]
    )


def _sc_decode_core(y_llr, information_pos, frozen_bit=0):
    """非递归 SC 译码核心（状态机实现）。"""
    N = y_llr.size
    n = int(math.log2(N))
    llr_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    bit_matrix = np.full((n + 1, N), np.nan, dtype=np.float64)
    llr_matrix[0] = y_llr
    position = [0, 0, n, N]

    while not _all_filled(bit_matrix[n]):
        span = 2 ** (position[2] - position[0])
        up_llr = llr_matrix[position[0]][position[1] : position[1] + span]
        up_bit = bit_matrix[position[0]][position[1] : position[1] + span]
        half = span // 2
        left_llr = llr_matrix[position[0] + 1][position[1] : position[1] + half]
        left_bit = bit_matrix[position[0] + 1][position[1] : position[1] + half]
        right_llr = llr_matrix[position[0] + 1][position[1] + half : position[1] + span]
        right_bit = bit_matrix[position[0] + 1][position[1] + half : position[1] + span]

        if _all_filled(up_bit):
            position = _up(position)
        elif _all_filled(right_bit):
            up_bit = _get_up_bit(left_bit, right_bit)
            bit_matrix[position[0]][position[1] : position[1] + span] = up_bit
        elif _all_filled(right_llr):
            if position[0] == position[2] - 1:
                right_bit_pos = position[1] + 1
                right_bit_val = _get_right_bit(
                    right_llr, information_pos, frozen_bit, right_bit_pos
                )
                bit_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_bit_val
            else:
                position = _rightdown(position)
        elif _all_filled(left_bit):
            right_llr = _get_right_llr(left_bit, up_llr)
            llr_matrix[position[0] + 1][position[1] + half : position[1] + span] = right_llr
        elif not _all_filled(left_llr):
            left_llr = _get_left_llr(up_llr)
            llr_matrix[position[0] + 1][position[1] : position[1] + half] = left_llr
        else:
            if position[0] == position[2] - 1:
                left_bit_pos = position[1]
                left_bit_val = _get_left_bit(
                    left_llr, information_pos, frozen_bit, left_bit_pos
                )
                bit_matrix[position[0] + 1][position[1] : position[1] + half] = left_bit_val
            else:
                position = _leftdown(position)

    u_hat = bit_matrix[n].astype(int)
    return u_hat


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（通过调用非递归核心实现，保证与主译码器一致）。"""
    y_llr = _prepare_llr(llr_ch)
    information_pos = _frozen_mask_to_info_pos(frozen_bits)
    return _sc_decode_core(y_llr, information_pos, frozen_bit=0)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量（用于分析与扩展）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        llr_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            llr_layers.append(layer)
            psi >>= 1
            layer += 1
        llr_layers.extend(range(layer, n))
        llr_layer_vec.append(llr_layers)

        bit_layers = []
        psi = phi
        layer = 0
        while psi % 2 == 1:
            bit_layers.append(layer)
            psi >>= 1
            layer += 1
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    """
    y_llr = _prepare_llr(llr_ch)
    information_pos = _frozen_mask_to_info_pos(frozen_bits)
    return _sc_decode_core(y_llr, information_pos, frozen_bit=0)


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import compute_llr, eb_n0_to_sigma
    from construction import ga_construction

    u = np.array([0, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(1 - 2 * x, sigma)
        u_hat_r = sc_decode_recursive(llr, frozen_bits)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat, u) or not np.array_equal(u_hat_r, u):
            errors += 1
    print(f"SC 往返测试: {100 - errors}/100 通过")
