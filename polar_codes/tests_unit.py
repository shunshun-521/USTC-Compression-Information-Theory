"""极化码模块单元测试。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, prepare_channel_llr, sc_decode_channel
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
  """运行所有单元测试，失败时抛出 AssertionError。"""
  # 编码器
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

  # GA 构造
  info8, frozen8, _ = ga_construction(8, 4, 2.5)
  assert np.array_equal(info8, [0, 3, 5, 6]), f"GA N=8 错误: {info8}"

  # CRC
  bits = crc_encode(np.array([1, 0, 1, 0, 1, 1, 0, 1]), 8)
  assert crc_check(bits, 8)
  bad = bits.copy()
  bad[0] ^= 1
  assert not crc_check(bad, 8)

  # SC 无损验证
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info_idx] = 0
  sigma = eb_n0_to_sigma(10.0, 0.5)
  rng = np.random.default_rng(123)
  for _ in range(100):
    u = np.zeros(N, dtype=int)
    u[info_idx] = rng.integers(0, 2, K)
    x = polar_encode(u)
    y = awgn_channel(bpsk_modulate(x), sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode_channel(llr, frozen)
    assert np.array_equal(u_hat, u), "SC 译码失败"

  # L=1 SCL 等价 SC
  llr_test = prepare_channel_llr(compute_llr(bpsk_modulate(polar_encode(u)), sigma))
  uh_sc = sc_decode(llr_test, frozen)
  uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(
    compute_llr(bpsk_modulate(polar_encode(u)), sigma)
  )
  assert np.array_equal(uh_sc, uh_scl), "L=1 SCL 与 SC 不一致"

  print("所有单元测试通过。")


if __name__ == "__main__":
  run_unit_tests()
