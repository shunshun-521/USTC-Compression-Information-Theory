"""极化码模块单元测试。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode, crc_check


def run_unit_tests():
  """运行所有单元测试，失败时抛出 AssertionError。"""
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

  assert np.array_equal(crc_encode([1, 0, 1, 1], 8)[-8:], crc_encode([1, 0, 1, 1], 8)[-8:])
  msg = np.array([1, 0, 1, 0, 1, 1, 0, 0])
  coded = crc_encode(msg, 8)
  assert crc_check(coded, 8)

  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info_idx] = 0
  sigma = eb_n0_to_sigma(10.0, K / N)
  rng = np.random.default_rng(0)
  for _ in range(100):
    u_full = np.zeros(N, dtype=int)
    u_full[info_idx] = rng.integers(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_full)) + rng.normal(0, sigma, N), sigma)
    u_sc = sc_decode(llr, frozen)
    u_rec = sc_decode_recursive(llr, frozen)
    assert np.array_equal(u_sc, u_rec), "非递归与递归 SC 不一致"
    assert np.array_equal(u_full, u_sc), "SC 高信噪比译码失败"

  scl = SCLDecoder(N, frozen, list_size=1)
  u_scl, _ = scl.decode(llr)
  assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 等价"

  print("所有单元测试通过。")


if __name__ == "__main__":
  run_unit_tests()
