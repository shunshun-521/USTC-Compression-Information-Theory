"""模块数值校验"""
import numpy as np

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_channel
from decoder_scl import SCLDecoder


def run_all():
    # G_N = B_N F^{⊗2}：u=[1,0,1,1] -> x=[1,0,1,1]（与蝶形+比特倒序一致）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
    u2 = np.array([0, 0, 1, 1])
    assert np.array_equal(polar_encode(u2), [0, 0, 1, 1])

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    rng = np.random.default_rng(0)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        payload = rng.integers(0, 2, size=K, dtype=np.int8)
        u_full = np.zeros(N, dtype=np.int8)
        u_full[info_idx] = payload
        x = polar_encode(u_full)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode_channel(llr, frozen)
        assert np.array_equal(u_hat[info_idx], payload)

    # 递归与非递归在译码域一致
    scl = SCLDecoder(N, frozen, list_size=1)
    u_scl, _ = scl.decode(llr)
    u_sc = sc_decode_channel(llr, frozen)
    assert np.array_equal(u_scl, u_sc), "L=1 SCL 应等价 SC"

    print("validate.py: all checks passed")


if __name__ == "__main__":
    run_all()
