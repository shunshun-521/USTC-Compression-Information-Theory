"""极化码模块单元测试（各实验脚本启动前调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def run_unit_tests(verbose=True):
    """运行全部校验，失败时抛出 AssertionError。"""
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # GA 构造 sanity
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    assert len(info8) == 4 and len(frozen8) == 4

    # SC 无损验证
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = np.random.randint(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen_bool)
        assert np.array_equal(u_hat[info_idx], payload), "SC 译码在 Eb/N0=10dB 下出错"

    # SCL L=1 等价 SC
    llr_test = compute_llr(bpsk_modulate(polar_encode(u)), 1e-3)
    scl = SCLDecoder(N, frozen_bool, list_size=1)
    u_scl, _ = scl.decode(llr_test)
    u_sc = sc_decode(llr_test, frozen_bool)
    assert np.array_equal(u_scl, u_sc), "SCL L=1 与 SC 不一致"

    # CRC
    info = np.random.randint(0, 2, 32)
    with_crc = crc_encode(info, 8)
    assert crc_check(with_crc, 8)

    # BP 低噪声
    bp = BPDecoder(N, frozen_bool, max_iter=50)
    payload = np.random.randint(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = payload
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    u_bp, _ = bp.decode(llr)
    assert np.array_equal(u_bp[info_idx], payload), "BP 无损译码失败"

    if verbose:
        print("全部单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
