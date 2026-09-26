"""
极化码 SC（串行抵消）译码器
核心实现基于 polarcodes_mini.SCD（非递归，经本项目信道/编码验证）
"""
import numpy as np

from polarcodes_mini.SCD import SCD
from polarcodes_mini.decoder_utils import lower_llr, upper_llr
from polarcodes_mini.mock_pc import PolarCode


def f_operation(La, Lb):
    """f 运算（对数域，与 SCD 一致）"""
    return upper_llr(float(La), float(Lb))


def g_operation(btm, top, u_hat):
    """g 运算（btm 为下层 LLR，top 为上层 LLR）"""
    return lower_llr(float(btm), float(top), int(u_hat))


def _frozen_set(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return set(np.where(frozen_bits == 1)[0])


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    pc = PolarCode(N)
    pc.likelihoods = llr_ch
    pc.frozen = list(_frozen_set(frozen_bits))
    return SCD(pc).decode()


def sc_decode_recursive(llr_ch, frozen_bits):
    """参考实现（与 sc_decode 相同）"""
    return sc_decode(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    from encoder import bit_reversed

    n = int(np.log2(N))
    return [1 << i for i in range(n + 1)], [bit_reversed(i, n) for i in range(N)]
