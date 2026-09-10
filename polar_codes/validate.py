"""模块正确性校验脚本"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def validate_encoder():
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert len(x) == 4
    x2 = polar_encode(u)
    assert np.array_equal(x, x2), "编码器应确定性输出"
    print(f"[PASS] 编码器: u={u} -> x={x}")


def validate_sc_decoder():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(123)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], payload):
            errors += 1
    assert errors == 0, f"SC 译码在 10dB 下有 {errors} 帧错误"
    print("[PASS] SC 译码无损验证 (N=64, K=32, 100帧)")


def validate_scl_equivalence():
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(456)
    mismatches = 0
    for _ in range(50):
        u = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u[info_idx] = payload
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(8.0, K / N))
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            mismatches += 1
    assert mismatches == 0, f"L=1 SCL 与 SC 有 {mismatches} 次不一致"
    print("[PASS] L=1 SCL 等价于 SC")


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
    encoded = crc_encode(bits, 8)
    assert crc_check(encoded, 8)
    encoded[-1] ^= 1
    assert not crc_check(encoded, 8)
    print("[PASS] CRC-8 编解码")


def run_all():
    validate_encoder()
    validate_sc_decoder()
    validate_scl_equivalence()
    validate_crc()
    info, frozen, _ = ga_construction(8, 4, 2.5)
    print(f"[INFO] N=8, K=4: info={info}, frozen={frozen}")
    info256, _, _ = ga_construction(256, 128, 2.5)
    print(f"[INFO] N=256, K=128, first 20 info: {info256[:20]}")


if __name__ == "__main__":
    run_all()
