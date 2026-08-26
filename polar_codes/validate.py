"""
单元测试与数值正确性校验
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode_with_llr_reversal
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert x.shape == (4,)
  u2 = np.array([0, 0, 1, 1])
  x2 = polar_encode(u2)
  assert np.array_equal(x2, [0, 0, 1, 1]), f"编码器自检失败: {x2}"


def test_ga_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  print("N=8, K=4, Eb/N0=2.5dB")
  print("info_indices:", info)
  print("frozen_indices:", frozen)
  info256, _, _ = ga_construction(256, 128, 2.5)
  print("N=256, K=128, first 20 info_indices:", info256[:20])


def test_sc_noiseless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(0)
  sigma = eb_n0_to_sigma(12.0, K / N)
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode_with_llr_reversal(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
      errors += 1
  assert errors == 0, f"SC 高信噪比测试失败: {errors}/100 帧有错误"


def test_scl_l1_equals_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  sigma = eb_n0_to_sigma(12.0, K / N)
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_sc = sc_decode_with_llr_reversal(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"


def test_crc():
  bits = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  coded = crc_encode(bits, 8)
  assert crc_check(coded, 8)
  coded[-1] ^= 1
  assert not crc_check(coded, 8)


def test_bp_short():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(2)
  sigma = eb_n0_to_sigma(8.0, K / N)
  bp = BPDecoder(N, frozen_bits, max_iter=50)
  errors = 0
  for _ in range(30):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat, _ = bp.decode(llr)
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
      errors += 1
  assert errors < 10, f"BP 短帧测试错误过多: {errors}/30"


def main():
  print("=" * 60)
  print("极化码模块校验")
  print("=" * 60)
  test_encoder()
  print("[PASS] 编码器")
  test_ga_construction()
  print("[PASS] GA 构造")
  test_crc()
  print("[PASS] CRC")
  test_sc_noiseless()
  print("[PASS] SC 高信噪比")
  test_scl_l1_equals_sc()
  print("[PASS] SCL(L=1) 等价 SC")
  test_bp_short()
  print("[PASS] BP 短帧")
  print("\n全部校验通过。")


if __name__ == "__main__":
  main()
