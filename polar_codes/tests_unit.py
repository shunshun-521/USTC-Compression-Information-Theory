"""极化码模块单元测试。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode, polar_generator_matrix


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  G = polar_generator_matrix(4)
  assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"


def test_ga_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  assert np.array_equal(info, [0, 3, 5, 6]), info
  info256, _, _ = ga_construction(256, 128, 2.5)
  expected = [1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38]
  assert np.array_equal(info256[:20], expected), info256[:20]


def test_sc_noiseless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(123)
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    sigma = eb_n0_to_sigma(10, K / N)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_hat = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_hat, u)


def test_scl_equals_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  scl = SCLDecoder(N, frozen_bits, list_size=1)
  rng = np.random.default_rng(7)
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_scl, _ = scl.decode(llr)
    u_sc = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_scl, u_sc)


def run_all():
  test_encoder()
  test_ga_construction()
  test_sc_noiseless()
  test_scl_equals_sc()
  print("All unit tests passed.")


if __name__ == "__main__":
  run_all()
