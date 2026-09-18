"""模块正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, generate_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_validation(verbose=True):
    """运行所有单元测试，返回是否全部通过"""
    passed = True

    def check(name, cond, msg=""):
        nonlocal passed
        if not cond:
            passed = False
            if verbose:
                print(f"FAIL [{name}]: {msg}")
        elif verbose:
            print(f"PASS [{name}]")

    N = 4
    G = generate_matrix(N)
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    x_mat = (u @ G) % 2
    check("encoder_matrix", np.array_equal(x, x_mat), f"x={x}, expected={x_mat}")

    for r in (8, 16):
        poly_len = r
        info = np.random.default_rng(0).integers(0, 2, 16)
        encoded = crc_encode(info, r)
        check(f"crc{r}", crc_check(encoded, r))
        bad = encoded.copy()
        bad[-1] ^= 1
        check(f"crc{r}_reject", not crc_check(bad, r))

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rng = np.random.default_rng(42)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            sc_errors += 1
    check("sc_noiseless_10db", sc_errors == 0, f"errors={sc_errors}/100")

    sigma_hi = eb_n0_to_sigma(15.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1)
    scl_errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma_hi, rng)
        llr = compute_llr(y, sigma_hi)
        u_hat, _ = scl.decode(llr)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            scl_errors += 1
    check("scl_L1_equiv_sc", scl_errors == 0, f"errors={scl_errors}/100")

    return passed


if __name__ == "__main__":
    ok = run_validation()
    print("\n全部通过" if ok else "\n存在失败项")
    raise SystemExit(0 if ok else 1)
