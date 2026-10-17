"""
极化码 SC（串行抵消）译码器
"""
import numpy as np
from _codec_core import f_hf as f_operation, g as g_operation, sc_decoder as _sc_decoder


def _info_indices(frozen_bits):
    return np.where(np.asarray(frozen_bits) == 0)[0]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 的别名：调用树形 SC 实现"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """保留接口：非递归索引预计算（当前译码器使用树形实现）"""
    n = int(np.log2(N))
    return [0] * (n + 1), [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码。
    frozen_bits: 1=冻结位，0=信息位
    """
    info_idx = _info_indices(frozen_bits)
    u_hat, _ = _sc_decoder(np.asarray(llr_ch, dtype=np.float64), info_idx, 0)
    return np.asarray(u_hat, dtype=int)
