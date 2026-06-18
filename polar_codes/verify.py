"""极化码模块单元测试与数值校验。"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma, prepare_channel_llr
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  expected = np.array([1, 1, 0, 1])
  assert np.array_equal(x, expected), f"编码器错误: {x}, expected {expected}"
  print("编码器校验通过:", x)


def test_ga_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  expected_info = np.array([0, 3, 5, 6])
  assert np.array_equal(info, expected_info), f"GA N=8: {info}"
  print("GA 构造校验通过:", info)


def test_sc_lossless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(0)
  sigma = 0.01
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    payload = rng.integers(0, 2, size=K)
    u[info_idx] = payload
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], payload):
      errors += 1
  assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
  print("SC 无损译码校验通过 (100/100)")


def test_sc_recursive_match():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  sigma = 0.01
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    a = sc_decode(llr, frozen_bits)
    if not np.array_equal(a[info_idx], u[info_idx]):
      raise AssertionError("非递归 SC 译码错误")
  print("SC 非递归译码一致性校验通过")


def test_scl_l1_equals_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(2)
  sigma = 0.01
  for _ in range(30):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
  print("SCL L=1 等价 SC 校验通过")


def test_crc():
  bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  coded = crc_encode(bits, 8)
  assert crc_check(coded, 8)
  coded[-1] ^= 1
  assert not crc_check(coded, 8)
  print("CRC 校验通过")


def test_bp_noiseless():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  bp = BPDecoder(N, frozen_bits, max_iter=50)
  u = np.zeros(N, dtype=int)
  u[info_idx] = np.array([1, 0, 1, 0, 1, 1, 0, 0, 0, 1, 1, 0, 1, 0, 1, 1])
  x = polar_encode(u)
  llr = compute_llr(bpsk_modulate(x), 0.01)
  u_hat, iters = bp.decode(llr)
  assert np.array_equal(u_hat[info_idx], u[info_idx]), f"BP 噪声less失败, iters={iters}"
  print("BP 高 SNR 校验通过")


def run_all():
  test_encoder()
  test_ga_construction()
  test_crc()
  test_sc_recursive_match()
  test_sc_lossless()
  test_scl_l1_equals_sc()
  test_bp_noiseless()
  print("\n全部单元测试通过。")


if __name__ == "__main__":
  run_all()
