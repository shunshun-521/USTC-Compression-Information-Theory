"""单元测试与模块验证"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
  """运行所有模块校验"""
  # 编码器校验：u=[0,0,1,1] 经 G_N 编码为 [0,0,1,1]
  u = np.array([0, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [0, 0, 1, 1]), f"编码器错误: {x}"

  # SC 译码校验（高信噪比）
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  sigma = eb_n0_to_sigma(12.0, K / N)
  rng = np.random.default_rng(123)
  errors = 0
  for _ in range(100):
    payload = rng.integers(0, 2, K)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = payload
    codeword = polar_encode(u_sent)
    y = awgn_channel(bpsk_modulate(codeword), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], payload):
      errors += 1
  assert errors == 0, f"SC 译码在高信噪比下失败: {errors}/100 帧错误"

  # 路径度量校验：L=1 的 SCL 应等价于 SC
  N, K = 32, 16
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(456)
  for _ in range(50):
    payload = rng.integers(0, 2, K)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = payload
    codeword = polar_encode(u_sent)
    llr = compute_llr(bpsk_modulate(codeword), eb_n0_to_sigma(8.0, 0.5))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

  # CRC 校验
  info = np.array([1, 0, 1, 1, 0, 1, 0, 0])
  coded = crc_encode(info, 8)
  assert crc_check(coded, 8)
  assert not crc_check(np.append(info, np.zeros(8, dtype=int)), 8)

  print("所有单元测试通过。")


if __name__ == '__main__':
  run_unit_tests()
