"""Unit tests for polar code modules."""
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from encoder import polar_encode
from construction import ga_construction
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder


def run_unit_tests():
  # 1. Encoder test
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
  print("✓ Encoder test passed")

  # 2. SC decoder lossless test
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=bool)
  frozen_bits[info_idx] = False

  rng = np.random.default_rng(0)
  sigma = eb_n0_to_sigma(10.0, K / N)
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
      errors += 1
  assert errors == 0, f"SC lossless test failed: {errors} errors"
  print("✓ SC lossless test passed")

  # 3. SCL L=1 equivalent to SC
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
  print("✓ SCL L=1 == SC test passed")

  # 4. Recursive vs non-recursive SC
  for _ in range(10):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u1 = sc_decode_recursive(llr, frozen_bits)
    u2 = sc_decode(llr, frozen_bits)
    assert np.array_equal(u1, u2), "Recursive != non-recursive SC"
  print("✓ Recursive == non-recursive SC test passed")

  print("\nAll unit tests passed!")


if __name__ == "__main__":
  run_unit_tests()
