"""
极化码模块数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from decoder_bp import BPDecoder


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x_butterfly = polar_encode(u)
  x_matrix = polar_encode_matrix(u)
  assert np.array_equal(x_butterfly, x_matrix), (
    f"编码器蝶形与矩阵不一致: {x_butterfly} vs {x_matrix}"
  )
  print(f"编码器校验通过: u={u} -> x={x_butterfly}")


def test_construction():
  info8, frozen8, _ = ga_construction(8, 4, 2.5)
  assert len(info8) == 4 and len(frozen8) == 4
  assert len(np.union1d(info8, frozen8)) == 8
  info256, _, _ = ga_construction(256, 128, 2.5)
  print(f"N=8 info={info8}, frozen={frozen8}")
  print(f"N=256 前20个信息位索引: {info256[:20]}")


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
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
      errors += 1
  assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"
  print("SC 无损译码校验通过 (N=64, K=32, Eb/N0=10dB, 100帧)")


def test_sc_recursive_match():
  N, K = 16, 8
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  sigma = eb_n0_to_sigma(5.0, K / N)
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
    llr = compute_llr(y, sigma)
    u_nr = sc_decode(llr, frozen_bits)
    u_rec = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u_nr, u_rec), "递归与非递归 SC 不一致"
  print("递归/非递归 SC 一致性校验通过")


def test_scl_l1_equals_sc():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(2)
  sigma = eb_n0_to_sigma(4.0, K / N)
  scl = SCLDecoder(N, frozen_bits, list_size=1)
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
  print("SCL L=1 等价 SC 校验通过")


def test_crc():
  info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  coded = crc_encode(info, 8)
  assert crc_check(coded, 8)
  assert not crc_check(np.concatenate([info, np.zeros(8, dtype=int)]), 8)
  print("CRC 校验通过")


def test_bp_single_frame():
  N, K = 16, 8
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  u = np.zeros(N, dtype=int)
  u[info_idx] = np.array([1, 0, 1, 1, 0, 1, 0, 1])
  x = polar_encode(u)
  sigma = eb_n0_to_sigma(8.0, K / N)
  y = awgn_channel(bpsk_modulate(x), sigma)
  llr = compute_llr(y, sigma)
  bp = BPDecoder(N, frozen_bits, max_iter=50)
  u_hat, iters = bp.decode(llr)
  assert u_hat[info_idx].tolist() == u[info_idx].tolist(), "BP 高信噪比单帧译码失败"
  print(f"BP 单帧译码校验通过 (iters={iters})")


def main():
  print("=" * 50)
  print("极化码模块校验")
  print("=" * 50)
  test_encoder()
  test_construction()
  test_crc()
  test_sc_recursive_match()
  test_sc_lossless()
  test_scl_l1_equals_sc()
  test_bp_single_frame()
  print("=" * 50)
  print("全部校验通过")
  print("=" * 50)


if __name__ == "__main__":
  main()
