"""
单元测试与数值校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, verify_sc_decoder
from decoder_scl import SCLDecoder, crc_encode, crc_check, verify_scl_equals_sc
from encoder import polar_encode


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
  u0 = np.zeros(4, dtype=int)
  assert np.array_equal(polar_encode(u0), np.zeros(4, dtype=int))


def test_construction():
  info8, frozen8, _ = ga_construction(8, 4, 2.5)
  print("N=8, K=4, Eb/N0=2.5dB")
  print("info_indices:", info8)
  print("frozen_indices:", frozen8)
  assert np.array_equal(info8, [0, 3, 5, 6])

  info256, _, _ = ga_construction(256, 128, 2.5)
  expected = np.array([1, 2, 4, 7, 8, 11, 13, 14, 16, 19, 21, 22, 25, 26, 28, 31, 32, 35, 37, 38])
  print("\nN=256, K=128, first 20 info_indices:", info256[:20])
  assert np.array_equal(info256[:20], expected)


def test_sc_noiseless():
  assert verify_sc_decoder(64, 32, 100, 12.0)


def test_scl_equals_sc():
  assert verify_scl_equals_sc(64, 50)


def test_crc():
  bits = np.array([1, 0, 1, 1, 0, 1, 0, 1], dtype=np.int8)
  coded = crc_encode(bits, 8)
  assert len(coded) == 16
  assert crc_check(coded, 8)


def test_bp_noiseless():
  N = 32
  K = 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=bool)
  frozen_bits[info_idx] = False
  bp = BPDecoder(N, frozen_bits, max_iter=50)
  ok = 0
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6) * 1e6
    u_hat, _ = bp.decode(llr)
    ok += np.array_equal(u[info_idx], u_hat[info_idx])
  assert ok >= 10, f"BP noiseless success rate too low: {ok}/20"


def main():
  np.random.seed(42)
  test_encoder()
  test_construction()
  test_sc_noiseless()
  test_scl_equals_sc()
  test_crc()
  test_bp_noiseless()
  print("\n所有校验通过。")


if __name__ == "__main__":
  main()
