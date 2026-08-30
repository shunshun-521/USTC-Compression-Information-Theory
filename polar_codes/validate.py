#!/usr/bin/env python3
"""极化码模块数值正确性校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, build_generator_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  G = build_generator_matrix(4)
  x_ref = (u @ G) % 2
  assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
  print("PASS: encoder")


def test_ga_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  expected_info = np.array([0, 3, 5, 6])
  assert np.array_equal(info, expected_info), f"GA N=8: {info}"
  print("PASS: GA construction N=8")


def test_sc_lossless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(0)
  sigma = eb_n0_to_sigma(10.0, K / N)
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u[info_idx], u_hat[info_idx]):
      errors += 1
  assert errors == 0, f"SC 无损译码失败: {errors}/100"
  print("PASS: SC lossless decode")


def test_sc_recursive_match():
  """递归 SC 参考实现冒烟测试。"""
  N = 16
  K = 8
  info_idx, _, _ = ga_construction(N, K, 2.0)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  sigma = eb_n0_to_sigma(8.0, 0.5)
  u = np.zeros(N, dtype=int)
  u[info_idx] = rng.integers(0, 2, K)
  llr = compute_llr(
    awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
  )
  _ = sc_decode_recursive(llr, frozen_bits)
  print("PASS: SC recursive smoke test")


def test_scl_equiv_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(2)
  sigma = eb_n0_to_sigma(8.0, K / N)
  for _ in range(30):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(
      awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
    )
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
  print("PASS: SCL L=1 == SC")


def test_crc():
  bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  encoded = crc_encode(bits, 8)
  assert crc_check(encoded, 8)
  encoded[-1] ^= 1
  assert not crc_check(encoded, 8)
  print("PASS: CRC")


if __name__ == "__main__":
  test_encoder()
  test_ga_construction()
  test_sc_lossless()
  test_sc_recursive_match()
  test_scl_equiv_sc()
  test_crc()
  print("\nAll validation tests passed.")
