"""模块正确性校验（各实验脚本运行前调用）。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import channel_llr_to_decoder, sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_validation(verbose=True):
  """运行编码器、SC/SCL 校验。"""
  # 编码器：标准蝶形 + 比特倒序
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  x_expected = np.array([1, 0, 1, 1])
  assert np.array_equal(x, x_expected), f"编码器错误: {x} != {x_expected}"

  u2 = np.array([0, 0, 1, 1])
  x2 = polar_encode(u2)
  assert np.array_equal(x2, [0, 0, 1, 1]), f"编码器手算校验失败: {x2}"

  # SC 无损译码
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=bool)
  frozen_bits[info_idx] = False
  sigma = eb_n0_to_sigma(15.0, K / N)

  sc_errors = 0
  for seed in range(100):
    rng = np.random.default_rng(seed)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    x_cod = polar_encode(u_sent)
    y = bpsk_modulate(x_cod) + rng.normal(0, sigma, N)
    llr = channel_llr_to_decoder(compute_llr(y, sigma))
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_sent[info_idx], u_hat[info_idx]):
      sc_errors += 1
  assert sc_errors == 0, f"SC 译码在 Eb/N0=15dB 失败 {sc_errors}/100 帧"

  # L=1 SCL 等价 SC
  for seed in range(20):
    rng = np.random.default_rng(seed + 1000)
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, K)
    x_cod = polar_encode(u_sent)
    y = bpsk_modulate(x_cod) + rng.normal(0, sigma, N)
    llr = channel_llr_to_decoder(compute_llr(y, sigma))
    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不等价"

  if verbose:
    print("所有模块校验通过。")
  return True


if __name__ == "__main__":
  run_validation()
