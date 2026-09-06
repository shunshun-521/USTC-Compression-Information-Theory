import numpy as np
from itertools import product

from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from decoder_sc import sc_decode
from channel import bpsk_modulate

u = np.array([1, 0, 1, 1])
x = polar_encode(u)
G = polar_generator_matrix(4)
assert np.array_equal(x, (u @ G) % 2), f"encoder error: {x}"

N = 8
info, _, _ = ga_construction(N, 4, 2.5)
fb = np.ones(N, dtype=bool)
fb[info] = False
errs = 0
for bits in product([0, 1], repeat=4):
    u = np.zeros(N, dtype=int)
    u[info] = list(bits)
    uh = sc_decode(100 * bpsk_modulate(polar_encode(u)), fb)
    if not np.array_equal(uh, u):
        errs += 1
print("N=8 exhaustive errs:", errs)
assert errs == 0

N, K = 64, 32
info, _, _ = ga_construction(N, K, 2.5)
fb = np.ones(N, dtype=bool)
fb[info] = False
rng = np.random.default_rng(0)
errs = 0
for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = 100 * bpsk_modulate(polar_encode(u))
    if not np.array_equal(sc_decode(llr, fb), u):
        errs += 1
print("N=64 noiseless errs:", errs)
assert errs == 0
print("ALL OK")
