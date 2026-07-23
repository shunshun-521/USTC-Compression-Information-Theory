import numpy as np
from encoder import polar_encode, polar_generator_matrix
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder

u = np.array([1, 0, 1, 1])
x = polar_encode(u)
G = polar_generator_matrix(4)
x_ref = (u @ G) % 2
assert np.array_equal(x, x_ref), f"encoder fail {x} {x_ref}"

N, K = 64, 32
info_idx, _, _ = ga_construction(N, K, 2.5)
frozen_bits = np.ones(N, dtype=int)
frozen_bits[info_idx] = 0
rng = np.random.default_rng(0)
sigma = eb_n0_to_sigma(15.0, K / N)
errors = 0
for _ in range(100):
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u_sent)
    s = bpsk_modulate(x)
    y = awgn_channel(s, sigma, rng)
    llr = compute_llr(y, sigma)
    uh = sc_decode(llr, frozen_bits)
    ur = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(uh, ur)
    if not np.array_equal(uh[info_idx], u_sent[info_idx]):
        errors += 1
assert errors == 0, f"SC errors={errors}"

llr_test = rng.normal(0, 1, N)
u_sc = sc_decode(llr_test, frozen_bits)
u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
assert np.array_equal(u_sc, u_scl)
print("ALL TESTS PASSED")
