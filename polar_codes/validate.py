"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import polar_encode, polar_encode_matrix


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  xm = polar_encode_matrix(u)
  assert np.array_equal(x, xm), f"编码器与矩阵法不一致: {x} vs {xm}"
  print("✓ 编码器校验通过")


def test_ga_construction():
  info8, frozen8, _ = ga_construction(8, 4, 2.5)
  assert len(info8) == 4 and len(frozen8) == 4
  info256, _, _ = ga_construction(256, 128, 2.5)
  print("✓ GA 构造校验通过")
  print(f"  N=8 info: {info8}, frozen: {frozen8}")
  print(f"  N=256 info (first 20): {info256[:20]}")


def test_sc_lossless():
  N, K = 64, 32
  design_ebn0 = 2.5
  info_idx, _, _ = ga_construction(N, K, design_ebn0)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(0)
  sigma = eb_n0_to_sigma(10.0, K / N)
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=np.int8)
    payload = rng.integers(0, 2, size=K, dtype=np.int8)
    u[info_idx] = payload
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], payload):
      errors += 1
  assert errors == 0, f"SC 无损测试失败，错误帧数={errors}"
  print("✓ SC 无损译码校验通过 (N=64, Eb/N0=10dB, 100帧)")


def test_sc_recursive_vs_nonrecursive():
  N, K = 128, 64
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  sigma = eb_n0_to_sigma(3.0, 0.5)
  for _ in range(20):
    u = np.zeros(N, dtype=np.int8)
    payload = rng.integers(0, 2, size=K, dtype=np.int8)
    u[info_idx] = payload
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 不一致"
  print("✓ SC 递归/非递归一致性校验通过")


def test_scl_equiv_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(2)
  sigma = eb_n0_to_sigma(4.0, 0.5)
  for _ in range(20):
    u = np.zeros(N, dtype=np.int8)
    payload = rng.integers(0, 2, size=K, dtype=np.int8)
    u[info_idx] = payload
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
  print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
  info = np.array([1, 0, 1, 1, 0, 0, 1, 0], dtype=np.int8)
  coded = crc_encode(info, 8)
  assert crc_check(coded, 8)
  bad = coded.copy()
  bad[0] ^= 1
  assert not crc_check(bad, 8)
  print("✓ CRC 校验通过")


def test_bp_roundtrip():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  u = np.zeros(N, dtype=np.int8)
  x = polar_encode(u)
  llr = compute_llr(bpsk_modulate(x), 0.01)
  u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
  assert np.array_equal(u_hat, u), "BP 无噪声全零码字译码失败"
  print(f"✓ BP 无噪声译码通过 (iters={iters})")


def run_all():
  print("=" * 50)
  print("极化码模块数值校验")
  print("=" * 50)
  test_encoder()
  test_ga_construction()
  test_crc()
  test_sc_lossless()
  test_sc_recursive_vs_nonrecursive()
  test_scl_equiv_sc()
  test_bp_roundtrip()
  print("=" * 50)
  print("全部测试通过！")
  print("=" * 50)


if __name__ == "__main__":
  run_all()
