#!/usr/bin/env python3
"""Quick SC decoder validation."""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma, awgn_channel, bit_reverse_llr
from decoder_sc import sc_decode

for N in [4, 8, 16]:
    K = N // 2
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    ok = 0
    for bits in range(2 ** K):
        u = np.zeros(N, dtype=int)
        u[info_idx] = [(bits >> i) & 1 for i in range(K)]
        llr = bit_reverse_llr(compute_llr(bpsk_modulate(polar_encode(u)), 0.1))
        ok += int(np.array_equal(u, sc_decode(llr, frozen)))
    print(f'N={N} noiseless {ok}/{2**K}')

N, K = 64, 32
info_idx, _, _ = ga_construction(N, K, 2.5)
frozen = np.ones(N, dtype=bool)
frozen[info_idx] = False
rng = np.random.default_rng(0)
err = 0
for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    sigma = eb_n0_to_sigma(10.0, K / N)
    llr = bit_reverse_llr(compute_llr(awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma))
    if not np.array_equal(u, sc_decode(llr, frozen)):
        err += 1
print(f'N=64 10dB err={err}/100')
