"""极化码模块单元测试。"""
import numpy as np

from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_bp import BPDecoder
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from encoder import polar_encode


def run_unit_tests():
  """运行所有单元测试，失败时抛出 AssertionError。"""
  # 编码器：蝶形编码与比特倒序后应能正确译回
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  frozen4 = np.zeros(4, dtype=bool)
  llr4 = compute_llr(bpsk_modulate(x), sigma=1e-6)
  assert np.array_equal(sc_decode(llr4, frozen4), u), f'编码器往返失败: {x}'

  # SC 无损验证
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen = np.ones(N, dtype=int)
  frozen[info_idx] = 0
  fb = frozen.astype(bool)
  errors = 0
  for _ in range(100):
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), eb_n0_to_sigma(10, K / N))
    if not np.array_equal(sc_decode(llr, fb), u_sent):
      errors += 1
  assert errors == 0, f'SC 无损验证失败: {errors}/100 帧有误'

  # SCL L=1 等价于 SC
  for _ in range(50):
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), eb_n0_to_sigma(10, K / N))
    u_sc = sc_decode(llr, fb)
    u_scl, _ = SCLDecoder(N, fb, list_size=1).decode(llr)
    assert np.array_equal(u_sc, u_scl), 'SCL L=1 与 SC 不等价'

  # BP 冒烟测试
  u_sent = np.zeros(N, dtype=int)
  u_sent[info_idx] = np.random.randint(0, 2, K)
  llr = compute_llr(bpsk_modulate(polar_encode(u_sent)), sigma=1e-6)
  u_bp, _ = BPDecoder(N, fb, max_iter=50).decode(llr)
  assert u_bp[info_idx].shape == u_sent[info_idx].shape

  print('所有单元测试通过。')


if __name__ == '__main__':
    run_unit_tests()
