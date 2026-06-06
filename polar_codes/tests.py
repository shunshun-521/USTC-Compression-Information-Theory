"""极化码模块单元测试"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def run_unit_tests():
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    # CRC
    bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 0]), 8)
    assert crc_check(bits, 8)

    # SC 无损验证（无噪信道）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)

    # 递归与非递归 SC 一致
    llr = rng.normal(size=N)
    u_rec = sc_decode_recursive(llr, frozen)
    u_fast = sc_decode(llr, frozen)
    assert np.array_equal(u_rec, u_fast)

    # L=1 SCL 等价 SC
    scl = SCLDecoder(N, frozen, list_size=1)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_scl, u_fast)

    # BP 冒烟测试
    bp = BPDecoder(N, frozen, max_iter=5)
    u_bp, _ = bp.decode(llr)
    assert u_bp.shape == (N,)

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
