"""单元测试：编码器与各译码器正确性校验"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode, bit_reversal_permutation
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def _matrix_G(N):
    F = np.array([[1, 0], [1, 1]], dtype=int)
    G = F.copy()
    n = int(np.log2(N))
    for _ in range(n - 1):
        G = np.kron(G, F) % 2
    br = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    B[np.arange(N), br] = 1
    return (B @ G) % 2


def run_unit_tests(verbose=True):
    ok = True

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G4 = _matrix_G(4)
    x_ref = (u @ G4) % 2
    if not np.array_equal(x, x_ref):
        ok = False
        if verbose:
            print(f"编码器与 G 矩阵不一致: {x} vs {x_ref}")

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info] = False
    br = bit_reversal_permutation(N)
    sigma = eb_n0_to_sigma(10.0, K / N)
    sc_ok = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        y = bpsk_modulate(x)
        llr = compute_llr(y, sigma)[br]
        uh = sc_decode(llr, frozen_bits)
        sc_ok += int(np.array_equal(uh, u))
    if sc_ok < 100:
        ok = False
        if verbose:
            print(f"SC 无损测试失败: {sc_ok}/100")

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info] = np.random.randint(0, 2, K)
        x = polar_encode(u)
        llr = compute_llr(bpsk_modulate(x), sigma)[br]
        uh_sc, _ = scl.decode(llr)
        uh_rec = sc_decode_recursive(llr, frozen_bits)
        if not np.array_equal(uh_sc, uh_rec):
            ok = False
            if verbose:
                print("L=1 SCL 与 SC 不一致")
            break

    if verbose:
        print("单元测试通过" if ok else "单元测试存在失败项")
    return ok


if __name__ == "__main__":
    run_unit_tests()
