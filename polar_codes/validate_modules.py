"""模块数值校验（各实验脚本导入前调用）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_construction():
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8, K=4, Eb/N0=2.5dB")
    print("info_indices:", info)
    print("frozen_indices:", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256, K=128, first 20 info_indices:", info256[:20])


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x} != {expected}"


def validate_sc():
    """N=64 噪声less + 高信噪比；长码长建议用 SCL 获得更低 FER。"""
    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        y = awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng)
        uh = sc_decode(compute_llr(y, sigma), frozen)
        if np.any(uh[info] != u[info]):
            errors += 1
    assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"


def validate_scl_vs_sc():
    N = 64
    info, _, _ = ga_construction(N, N // 2, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    u = np.zeros(N, dtype=int)
    u[info] = np.random.randint(0, 2, len(info))
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    assert np.array_equal(uh_sc, uh_scl), "SCL L=1 与 SC 不等价"


def validate_crc():
    msg = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    coded = crc_encode(msg, 8)
    assert crc_check(coded, 8)
    assert not crc_check(np.flip(coded), 8)


def run_all():
    validate_construction()
    validate_encoder()
    validate_sc()
    validate_scl_vs_sc()
    validate_crc()
    print("全部模块校验通过。")


if __name__ == "__main__":
    run_all()
