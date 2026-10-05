"""模块级单元测试（仿真脚本启动时调用）"""
import numpy as np
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, sc_decode_nonrecursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import bit_reversal_permutation


def run_validations(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 0, 1, 1])
    if not np.array_equal(x, expected):
        ok = False
        if verbose:
            print(f"[FAIL] 编码器: 得到 {x}, 期望 {expected}")

    N = 64
    K = 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info] = 0
    br = bit_reversal_permutation(N)
    rng = np.random.default_rng(123)
    sc_err = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), eb_n0_to_sigma(10.0, K / N))
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info], u[info]):
            sc_err += 1
    if sc_err > 0:
        ok = False
        if verbose:
            print(f"[FAIL] SC 无损测试: {sc_err}/100 帧错误")
    elif verbose:
        print("[OK] SC 无损测试 (N=64, Eb/N0=10dB)")

    llr = compute_llr(bpsk_modulate(polar_encode(np.zeros(N, int))), eb_n0_to_sigma(10.0, K / N))
    u_sc = sc_decode(llr, frozen)
    dec_l1 = SCLDecoder(N, frozen, list_size=1, crc_length=0)
    u_scl, _ = dec_l1.decode(llr)
    if not np.array_equal(u_sc, u_scl):
        ok = False
        if verbose:
            print("[FAIL] SCL L=1 与 SC 不一致")
    elif verbose:
        print("[OK] SCL L=1 等价 SC")

    msg = np.array([1, 0, 1, 0, 1, 1, 0, 1])
    payload = crc_encode(msg, 8)
    if not crc_check(payload, 8):
        ok = False
        if verbose:
            print("[FAIL] CRC-8 校验")

    info8, _, _ = ga_construction(8, 4, 2.5)
    frozen8 = np.ones(8, dtype=int)
    frozen8[info8] = 0
    if verbose:
        print("N=8 GA info:", info8, "frozen:", np.where(frozen8)[0])
        info256, _, _ = ga_construction(256, 128, 2.5)
        print("N=256 info (first 20):", info256[:20])

    return ok


if __name__ == "__main__":
    success = run_validations()
    raise SystemExit(0 if success else 1)
