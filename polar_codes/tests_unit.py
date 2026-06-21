"""极化码模块单元测试"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
  assert np.array_equal(x, polar_encode_matrix(u)), "蝶形编码与矩阵编码不一致"


def test_ga_construction():
  info8, frozen8, _ = ga_construction(8, 4, 2.5)
  assert np.array_equal(info8, [0, 3, 5, 6])
  assert np.array_equal(frozen8, [1, 2, 4, 7])


def test_sc_noiseless():
  N, K = 64, 32
  info, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info] = 0
  rng = np.random.default_rng(0)
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
    u_hat = sc_decode(llr, frozen.astype(bool))
    if not np.array_equal(u_hat, u):
      errors += 1
  assert errors <= 2, f"SC 译码在低噪声下出现 {errors}/100 错误"


def test_scl_l1_equals_sc():
  N, K = 32, 16
  info, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info] = 0
  rng = np.random.default_rng(1)
  sigma = eb_n0_to_sigma(4.0, 0.5)
  mism = 0
  for _ in range(30):
    u = np.zeros(N, dtype=int)
    u[info] = rng.integers(0, 2, K)
    y = bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen.astype(bool))
    u_scl, _ = SCLDecoder(N, frozen.astype(bool), list_size=1).decode(llr)
    if not np.array_equal(u_sc, u_scl):
      mism += 1
  assert mism == 0, f"L=1 SCL 与 SC 不一致: {mism} 帧"


def test_crc():
  bits = np.array([1, 0, 1, 0, 1, 1, 0, 1])
  coded = crc_encode(bits, 8)
  assert crc_check(coded, 8)
  assert not crc_check(np.append(bits, [1, 0, 0, 0, 0, 0, 0, 1]), 8)


def run_all():
  test_encoder()
  test_ga_construction()
  test_sc_noiseless()
  test_scl_l1_equals_sc()
  test_crc()
  print("All unit tests passed.")


if __name__ == "__main__":
  run_all()
