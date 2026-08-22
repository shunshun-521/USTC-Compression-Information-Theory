"""
极化码参考实现封装（基于 polar-codes 库核心模块）
"""
import os
import sys

import numpy as np

_REF = os.path.join(os.path.dirname(__file__), "polar_ref")
if _REF not in sys.path:
    sys.path.insert(0, _REF)

from Construct import Construct  # noqa: E402
from Encode import Encode  # noqa: E402
from SCD import SCD  # noqa: E402


class _PolarCode:
    """最小 PolarCode 对象，供 polar-codes 模块使用"""

    def __init__(self, N, K, construction="ga"):
        self.N = N
        self.M = N
        self.K = K
        self.n = int(np.log2(N))
        self.u = np.zeros(N, dtype=int)
        self.x = np.zeros(N, dtype=int)
        self.frozen = np.array([], dtype=int)
        self.frozen_lookup = np.ones(N, dtype=int)
        self.likelihoods = np.zeros(N, dtype=np.float64)
        self.message = np.zeros(K, dtype=int)
        self.construction_type = construction
        self.T = None
        self.F = None

    def get_normalised_SNR(self, design_snr_db):
        return 10.0 ** (design_snr_db / 10.0) * (self.K / self.M)

    def get_lut(self, my_set):
        lut = np.ones(self.N, dtype=int)
        for idx in my_set:
            lut[int(idx)] = 0
        return lut

    def set_message(self, m, info_lookup):
        self.message = np.asarray(m, dtype=int)
        self.x = np.zeros(self.N, dtype=int)
        self.x[info_lookup == 1] = self.message
        self.u = self.x.copy()


def construct_code(N, K, design_eb_n0_db, method="ga"):
    """构造极化码，返回 pc 对象及 info/frozen 索引"""
    pc = _PolarCode(N, K, "ga" if method == "ga" else "bb")
    Construct(pc, design_eb_n0_db)
    pc.frozen_lookup = pc.get_lut(pc.frozen)
    info_indices = np.where(pc.frozen_lookup == 1)[0]
    frozen_indices = np.sort(pc.frozen)
    return pc, info_indices, frozen_indices


def encode_block(pc, info_bits, info_lookup):
    """编码信息比特"""
    pc.set_message(info_bits, info_lookup)
    Encode(pc)
    return pc.u.copy()


def decode_sc(llr, pc):
    """SC 译码"""
    pc.likelihoods = np.asarray(llr, dtype=np.float64)
    return SCD(pc).decode()
