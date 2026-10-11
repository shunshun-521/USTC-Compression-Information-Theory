"""单元测试与数值校验"""
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, sc_decoders_match
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, prepare_channel_llr


def run_unit_tests(verbose=True):
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: u={u}, x={x}"  # 蝶形编码
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 1, 0, 1]), f"编码器错误: u2={u2}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    rng = np.random.default_rng(0)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u_sent = np.zeros(N, dtype=np.int8)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(prepare_channel_llr(llr, N), frozen_bits)
        assert np.array_equal(u_hat[info_idx], payload)

    llr_test = prepare_channel_llr(rng.normal(0, 1, size=64), N)
    assert sc_decoders_match(llr_test, frozen_bits)

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    u_scl, _ = scl.decode(llr_test)
    u_sc = sc_decode(llr_test, frozen_bits)
    assert np.array_equal(u_scl, u_sc), "L=1 SCL 应等价于 SC"

    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    coded = crc_encode(bits[:4], 8)
    assert crc_check(coded, 8)
    assert not crc_check(coded[:-1], 8)

    if verbose:
        print("validate.py: 所有单元测试通过。")
    return True


if __name__ == "__main__":
    run_unit_tests()
