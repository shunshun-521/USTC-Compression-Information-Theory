"""模块数值校验（实验脚本启动前调用）"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_validation(verbose=True):
    u = np.array([1, 0, 1, 1], dtype=np.int8)
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1], dtype=np.int8)
    assert np.array_equal(x, expected), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=np.int8)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, 0.5))
        if not np.array_equal(u, sc_decode(llr, frozen)):
            sc_err += 1
    assert sc_err == 0, f"SC 低噪声校验失败: {sc_err}/100"

    u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(
        compute_llr(bpsk_modulate(polar_encode(np.zeros(N, dtype=np.int8))), 0.1)
    )
    assert u_scl is not None

    info = np.array([1, 0, 1, 0, 1, 1], dtype=np.int8)
    crc = crc_encode(info, 8)
    assert crc_check(crc, 8)

    if verbose:
        info8, frozen8, _ = ga_construction(8, 4, 2.5)
        print("GA N=8,K=4 info:", info8, "frozen:", frozen8)
        info256, _, _ = ga_construction(256, 128, 2.5)
        print("GA N=256,K=128 info first 20:", info256[:20])
        print("validate.py: 全部校验通过")


if __name__ == "__main__":
    run_validation()
