"""极化码模块单元测试与校验。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, channel_llr_to_decoder
from channel import bpsk_modulate, compute_llr
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    """运行所有模块校验，失败时抛出 AssertionError。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    llr = channel_llr_to_decoder(compute_llr(bpsk_modulate(x), 0.001))
    uh = sc_decode(llr, np.zeros(4, dtype=int))
    assert np.array_equal(uh, u), f"N=4 SC 往返失败: u={u}, x={x}, uh={uh}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    errors = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        codeword = polar_encode(u)
        llr = channel_llr_to_decoder(compute_llr(bpsk_modulate(codeword), 0.001))
        uh = sc_decode(llr, frozen_bits)
        if not np.array_equal(uh, u):
            errors += 1
    assert errors == 0, f"SC 高 SNR 测试失败: {errors}/100 帧有错"

    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    llr_test = channel_llr_to_decoder(
        compute_llr(bpsk_modulate(polar_encode(u_test)), 0.01)
    )
    uh_sc = sc_decode(llr_test, frozen_bits)
    uh_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 与 SC 不一致"

    print("All unit tests passed.")


if __name__ == "__main__":
    run_unit_tests()
