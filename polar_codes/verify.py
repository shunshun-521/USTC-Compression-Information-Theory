"""极化码模块单元测试与数值校验"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import build_generator_matrix, polar_encode


def test_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  G = build_generator_matrix(4)
  expected = (u @ G) % 2
  assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"
  print("编码器校验通过:", x.tolist())


def test_ga_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  assert np.array_equal(info, [0, 3, 5, 6]), info
  info256, _, _ = ga_construction(256, 128, 2.5)
  print("GA N=8 info:", info)
  print("GA N=256 info (前20):", info256[:20])


def test_sc_noiseless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info_idx] = 0
  rng = np.random.default_rng(123)
  for _ in range(100):
    bits = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = bits
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    assert np.array_equal(sc_decode(llr, frozen), u)
    assert np.array_equal(sc_decode_recursive(llr, frozen), u)
  print("SC 无损校验通过 (100 帧)")


def test_scl_l1_equals_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info_idx] = 0
  rng = np.random.default_rng(7)
  scl = SCLDecoder(N, frozen, list_size=1)
  for _ in range(20):
    bits = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = bits
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u_sc = sc_decode(llr, frozen)
    u_scl, _ = scl.decode(llr)
    assert np.array_equal(u_sc, u_scl)
  print("SCL L=1 等价 SC 校验通过")


def test_crc():
  bits = np.array([1, 0, 1, 1, 0, 1, 0, 1])
  coded = crc_encode(bits, 8)
  assert crc_check(coded, 8)
  print("CRC-8 校验通过")


def test_bp_noiseless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info_idx] = 0
  bp = BPDecoder(N, frozen, max_iter=50)
  rng = np.random.default_rng(9)
  u = np.zeros(N, dtype=int)
  u[info_idx] = rng.integers(0, 2, K)
  x = polar_encode(u)
  llr = compute_llr(bpsk_modulate(x), 0.01)
  u_hat, iters = bp.decode(llr)
  assert np.array_equal(u_hat, u)
  print(f"BP 无损校验通过 (迭代={iters})")


def run_all():
  test_encoder()
  test_ga_construction()
  test_crc()
  test_sc_noiseless()
  test_scl_l1_equals_sc()
  test_bp_noiseless()
  print("\n全部单元测试通过。")


if __name__ == "__main__":
  run_all()
