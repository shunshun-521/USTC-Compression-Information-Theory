"""SC 译码对外接口"""
import numpy as np

from polar_scd import PolarSCD
from polar_scd_utils import bit_reversed, lower_llr, upper_llr


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def precompute_sc_indices(N):
    n = int(np.log2(N))
    return list(range(N)), [[bit_reversed(i, n) for i in range(N)]], [[]] * N


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_idx = np.where(frozen_bits)[0]
    return PolarSCD(len(llr_ch), llr_ch).decode(frozen_idx)


def sc_decode_recursive(llr_ch, frozen_bits):
    return sc_decode_nonrecursive(llr_ch, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_nonrecursive(llr_ch, frozen_bits)


class _SCDEngine(PolarSCD):
    """SCL 复用引擎（别名）"""

    def decode(self, frozen_set):
        return super().decode(np.array(list(frozen_set), dtype=np.int64))
