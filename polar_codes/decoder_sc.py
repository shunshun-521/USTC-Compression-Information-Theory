"""
极化码 SC（串行抵消）译码器
核心算法参考 PolarCodesPython（sc_decoder 状态机实现）
"""
import numpy as np
from sc_decoder_impl import sc_decoder


def f_operation(La, Lb):
    from sc_functions import f_hf

    return f_hf(La, Lb)


def g_operation(La, Lb, u_hat):
    from sc_functions import g

    return g(La, Lb, u_hat)


def precompute_sc_indices(N):
    """SCL 使用的占位索引（SCL 独立实现）"""
    import math

    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 2 ** (i - 1)
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode_recursive(llr, frozen_bits):
    info_pos = np.where(np.asarray(frozen_bits, dtype=int) == 0)[0].tolist()
    frozen_val = 0
    return sc_decoder(np.asarray(llr, dtype=np.float64), info_pos, frozen_val)[0].astype(int)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_recursive_channel(llr_ch, frozen_bits):
    return sc_decode(llr_ch, frozen_bits)
