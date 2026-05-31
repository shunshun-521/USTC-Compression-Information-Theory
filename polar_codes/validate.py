"""模块数值正确性校验。"""
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, scl_decode_equivalent_sc
from construction import ga_construction


def run_unit_tests(verbose=True):
    """运行全部单元测试，失败则抛出 AssertionError。"""
    # 编码器：与 GA-极化标准蝶形编码一致（往返验证）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4, f"编码器长度错误: {x}"

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(123)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
        if not np.array_equal(sc_decode(llr, frozen_bits), u):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码失败 {sc_errors}/100 帧"

    # SCL L=1 等价 SC
    llr_test = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)
    frozen4 = np.zeros(4, dtype=bool)
    uh_sc = sc_decode(llr_test, frozen4)
    uh_scl = scl_decode_equivalent_sc(llr_test, frozen4)
    assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不一致"

    if verbose:
        print("全部单元测试通过。")
    return True
