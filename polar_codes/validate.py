"""模块数值校验（实验脚本启动前调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode
from simulation import run_simulation


def run_all():
    # 编码器
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    # CRC
    msg = np.array([1, 0, 1, 1, 0, 0, 1, 1])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)

    # SC 高信噪比
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    rng = np.random.default_rng(123)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), 1e-3)
        assert np.array_equal(sc_decode(llr, frozen), u)

    # SCL L=1 等价 SC
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 1e-3)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl)

    print("validate.py: 所有校验通过。")


if __name__ == "__main__":
    run_all()
