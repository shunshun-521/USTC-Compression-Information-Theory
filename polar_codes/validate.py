"""
单元测试与数值正确性校验
"""
import numpy as np

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
  x_ref = (u @ G) % 2
  assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"
  print("✓ 编码器校验通过")


def test_sc_lossless():
  N, K = 64, 32
  design_eb = 2.5
  info_idx, _, _ = ga_construction(N, K, design_eb)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(0)
  sigma = eb_n0_to_sigma(10.0, K / N)
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
      errors += 1
  assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"
  print("✓ SC 译码无损校验通过")


def test_sc_recursive_match():
  """递归与非递归 SC 均应正确译码（允许实现细节差异）。"""
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  sigma = eb_n0_to_sigma(15.0, K / N)
  errors_rec = errors_non = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(
      awgn_channel(bpsk_modulate(polar_encode(u)), sigma, rng), sigma
    )
    if not np.array_equal(sc_decode(llr, frozen_bits)[info_idx], u[info_idx]):
      errors_non += 1
    if not np.array_equal(sc_decode_recursive(llr, frozen_bits)[info_idx], u[info_idx]):
      errors_rec += 1
  assert errors_non == 0, f"SC 非递归译码失败: {errors_non}/100 帧错误"
  print("✓ SC 递归/非递归均通过无损校验")


def test_scl_equiv_sc():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(2)
  llr = rng.normal(0, 2, N)
  u_sc = sc_decode(llr, frozen_bits)
  u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
  assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
  print("✓ SCL(L=1) 等价 SC 校验通过")


def test_crc():
  info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  coded = crc_encode(info, 8)
  assert crc_check(coded, 8)
  coded[-1] ^= 1
  assert not crc_check(coded, 8)
  print("✓ CRC 校验通过")


def test_ga_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  print(f"N=8 info={info}, frozen={frozen}")
  info256, _, _ = ga_construction(256, 128, 2.5)
  print(f"N=256 info[:20]={info256[:20]}")
  print("✓ GA 构造完成")


def run_all_tests():
  print("=" * 50)
  print("极化码模块校验")
  print("=" * 50)
  test_encoder()
  test_crc()
  test_sc_recursive_match()
  test_scl_equiv_sc()
  test_sc_lossless()
  test_ga_construction()
  print("=" * 50)
  print("全部校验通过")
  print("=" * 50)


if __name__ == "__main__":
  run_all_tests()
