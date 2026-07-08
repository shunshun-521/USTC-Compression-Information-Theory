"""单元测试与模块校验。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, verify_sc_decoders
from decoder_scl import verify_scl_equals_sc, crc_encode, crc_check


def run_unit_tests():
    """运行所有模块单元测试，失败时抛出 AssertionError。"""
    # 编码器校验
    N = 4
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = np.kron(F, F)
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i in range(N):
        B[i, br[i]] = 1
    GN = (G @ B) % 2

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_ref = np.mod(u @ GN, 2)
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"

    # GA 构造校验
    info8, frozen8, _ = ga_construction(8, 4, 2.5)
    print(f"N=8, K=4: info={info8}, frozen={frozen8}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"N=256, K=128, info (first 20): {info256[:20]}")

    # SC 译码校验
    verify_sc_decoders(N=64, K=32, num_frames=100, eb_n0_db=10.0)

    # SCL L=1 等价 SC
    verify_scl_equals_sc(N=64, K=32, num_frames=50, eb_n0_db=5.0)

    # CRC 校验
    info = np.random.randint(0, 2, 16)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)

    print("所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
