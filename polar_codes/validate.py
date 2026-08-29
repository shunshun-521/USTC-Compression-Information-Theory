#!/usr/bin/env python3
"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, build_generator_matrix


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  G = build_generator_matrix(4)
  x_mat = (u @ G) % 2
  assert np.array_equal(x, x_mat), f"编码器错误: {x} vs {x_mat}"
  print(f"[PASS] 编码器: polar_encode([1,0,1,1]) = {x}")


def test_ga_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  assert len(info) == 4 and len(frozen) == 4
  print(f"[PASS] GA N=8: info={info}, frozen={frozen}")

  info256, _, _ = ga_construction(256, 128, 2.5)
  print(f"[PASS] GA N=256 前20个信息位: {info256[:20]}")


def test_sc_noiseless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(0)
  sigma = eb_n0_to_sigma(10.0, K / N)
  errors = 0

  for _ in range(100):
    info = rng.integers(0, 2, size=K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], info):
      errors += 1

  assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧错误"
  print("[PASS] SC 译码高信噪比 100 帧无错误")


def test_sc_recursive_match():
  N = 16
  info_idx, _, _ = ga_construction(N, 8, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  llr = rng.normal(0, 5, size=N)

  u1 = sc_decode(llr, frozen_bits)
  u2 = sc_decode_recursive(llr, frozen_bits)
  assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
  print("[PASS] 递归与非递归 SC 一致")


def test_scl_equals_sc():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(2)
  sigma = eb_n0_to_sigma(4.0, K / N)

  for _ in range(20):
    info = rng.integers(0, 2, size=K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

  print("[PASS] L=1 SCL 等价于 SC")


def test_crc():
  info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  coded = crc_encode(info, 8)
  assert crc_check(coded, 8)
  coded_bad = coded.copy()
  coded_bad[-1] ^= 1
  assert not crc_check(coded_bad, 8)
  print("[PASS] CRC-8 编解码")


def main():
  test_encoder()
  test_ga_construction()
  test_sc_noiseless()
  test_sc_recursive_match()
  test_scl_equals_sc()
  test_crc()
  print("\n所有单元测试通过。")


if __name__ == "__main__":
  main()
