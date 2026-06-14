"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"


def test_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  assert len(info) == 4 and len(frozen) == 4
  info256, _, _ = ga_construction(256, 128, 2.5)
  print("N=8 info:", info)
  print("N=256 first20:", info256[:20])


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
    y = bpsk_modulate(x) + np.random.default_rng().normal(0, sigma, N)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u[info_idx], u_hat[info_idx]):
      errors += 1
  assert errors == 0, f"SC 无损测试失败: {errors}/100"


def test_scl_equals_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  sigma = eb_n0_to_sigma(6.0, K / N)
  mism = 0
  for _ in range(30):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = bpsk_modulate(x) + rng.normal(0, sigma, N)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
      mism += 1
  assert mism == 0, f"SCL(L=1) 与 SC 不一致: {mism}/30"


def test_crc():
  bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  coded = crc_encode(bits, 8)
  assert crc_check(coded, 8)


def test_bp():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  u = np.zeros(N, dtype=int)
  u[info_idx] = 1
  x = polar_encode(u)
  llr = (1 - 2 * x) * 20.0
  u_hat, iters = BPDecoder(N, frozen_bits).decode(llr)
  assert iters >= 1


if __name__ == "__main__":
  test_encoder()
  test_construction()
  test_sc_lossless()
  test_scl_equals_sc()
  test_crc()
  test_bp()
  print("All tests passed.")
