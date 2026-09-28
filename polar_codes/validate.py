"""单元测试与快速校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def validate_all():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, np.array([1, 0, 1, 1])), f"编码器错误: {x}"

    N, K = 4, 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    for mask in range(1 << K):
        u = np.zeros(N, dtype=int)
        u[info_idx] = [(mask >> i) & 1 for i in range(K)]
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        uh_sc = sc_decode(llr, frozen)
        assert np.array_equal(uh_sc, u), f"SC 无损校验失败: u={u}, uh={uh_sc}"

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.array([0, 1])
    x = polar_encode(u)
    llr = np.where(x == 0, 100.0, -100.0)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_scl, uh_sc), "L=1 SCL 应等价 SC"

    print("validate_all: OK")


if __name__ == "__main__":
    validate_all()
