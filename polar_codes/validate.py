"""
单元测试与模块验证
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_check, crc_encode
from encoder import build_generator_matrix, polar_encode


def validate_encoder():
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  G = build_generator_matrix(4)
  x_mat = (u @ G) % 2
  assert np.array_equal(x, x_mat), f"编码器与生成矩阵不一致: {x} vs {x_mat}"
  assert np.array_equal(x, [1, 1, 0, 1]), f"编码器参考向量错误: {x}"
  print("✓ 编码器校验通过")


def validate_sc_lossless():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  rng = np.random.default_rng(0)
  sigma = 1e-6
  errors = 0
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    x = polar_encode(u)
    y = bpsk_modulate(x)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], u[info_idx]):
      errors += 1
  assert errors == 0, f"SC 无损译码失败，错误帧数={errors}"
  print("✓ SC 无损译码校验通过")


def validate_sc_recursive():
  N = 16
  info_idx, _, _ = ga_construction(N, 8, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(1)
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=8)
    x = polar_encode(u)
    llr = compute_llr(bpsk_modulate(x), 0.01)
    u1 = sc_decode(llr, frozen_bits)
    u2 = sc_decode_recursive(llr, frozen_bits)
    assert np.array_equal(u1, u2), "递归与非递归 SC 结果不一致"
  print("✓ SC 递归/非递归一致性校验通过")


def validate_scl_equiv_sc():
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(2)
  for _ in range(20):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, size=K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.05)
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"
  print("✓ SCL(L=1) 等价 SC 校验通过")


def validate_crc():
  info = np.array([1, 0, 1, 1, 0, 0, 1, 0])
  coded = crc_encode(info, 8)
  assert crc_check(coded, 8)
  coded[-1] ^= 1
  assert not crc_check(coded, 8)
  print("✓ CRC 校验通过")


def validate_construction():
  info, frozen, _ = ga_construction(8, 4, 2.5)
  print(f"N=8 构造 info={info}, frozen={frozen}")
  info256, _, _ = ga_construction(256, 128, 2.5)
  print(f"N=256 前20个信息位索引: {info256[:20]}")


def validate_bp():
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  u = np.zeros(N, dtype=int)
  u[info_idx] = np.array([1, 0, 1, 1, 0, 0, 1, 0, 1, 1, 0, 1, 0, 0, 1, 1])
  llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.01)
  u_hat, iters = BPDecoder(N, frozen_bits, max_iter=50).decode(llr)
  assert np.array_equal(u_hat[info_idx], u[info_idx]), "BP 无噪声译码失败"
  print(f"✓ BP 无噪声译码通过 (iters={iters})")


def run_all():
  print("=" * 50)
  print("极化码模块验证")
  print("=" * 50)
  validate_encoder()
  validate_sc_recursive()
  validate_sc_lossless()
  validate_scl_equiv_sc()
  validate_crc()
  validate_bp()
  validate_construction()
  print("=" * 50)
  print("全部验证通过")
  print("=" * 50)


if __name__ == "__main__":
  run_all()
