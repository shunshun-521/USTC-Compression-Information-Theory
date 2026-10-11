"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
from numpy.typing import NDArray

from decoder_scl import scl_decode


def sc_decode(llr_ch, frozen_bits):
    """SC 译码（SCL 列表大小为 1）"""
    frozen = _to_bool_frozen(frozen_bits)
    paths = scl_decode(llr_ch, frozen, list_size=1)
    return paths[0][1].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """与 sc_decode 等价（保留接口）"""
    return sc_decode(llr_ch, frozen_bits)


def f_operation(La, Lb):
    sign = np.where(La >= 0, 1.0, -1.0) * np.where(Lb >= 0, 1.0, -1.0)
    return sign * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, [[] for _ in range(N)], [[] for _ in range(N)]


def _to_bool_frozen(frozen_bits):
    fb = np.asarray(frozen_bits)
    if fb.dtype == bool:
        return fb
    return fb.astype(int) != 0
