"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from encoder import polar_encode, polar_encode_matrix
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode


def _quick_mode():
  return os.environ.get("POLAR_QUICK", "").strip() in ("1", "true", "yes")


def run_unit_tests():
  """数值正确性校验"""
  # 编码器校验（与生成矩阵一致）
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  G = polar_encode_matrix(4)
  x_expected = (u @ G) % 2
  assert np.array_equal(x, x_expected), f"编码器错误: {x} != {x_expected}"

  # SC 译码校验（极低噪声）
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(123)
  sigma = eb_n0_to_sigma(10.0, 0.5)
  n_frames = 20 if _quick_mode() else 100
  for _ in range(n_frames):
    u_sent = np.zeros(N, dtype=int)
    u_sent[info_idx] = rng.integers(0, 2, len(info_idx))
    x = polar_encode(u_sent)
    y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    assert np.array_equal(u_hat[info_idx], u_sent[info_idx]), "SC 译码失败"

  # 非递归与递归 SC 一致
  llr_test = rng.normal(0, 1, N)
  assert np.array_equal(
    sc_decode(llr_test, frozen_bits),
    sc_decode_recursive(llr_test, frozen_bits),
  ), "SC 递归/非递归不一致"

  # L=1 SCL 等价于 SC
  scl = SCLDecoder(N, frozen_bits, list_size=1)
  u_scl, _ = scl.decode(llr_test)
  assert np.array_equal(u_scl, sc_decode(llr_test, frozen_bits)), "L=1 SCL 与 SC 不一致"

  print("所有单元测试通过。")


def run_simulation(
    N, K, eb_n0_db_list, decoder,
    decoder_type="sc",
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
):
  """
  蒙特卡洛仿真。
  """
  if _quick_mode():
    max_frames = min(max_frames, 500)
    min_errors = min(min_errors, 10)

  rng = np.random.default_rng(seed)
  rate = K / N
  K_info = K - crc_length

  if info_indices is None:
    info_indices, _, _ = ga_construction(N, K, 2.5)

  results = []

  for eb_n0_db in eb_n0_db_list:
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    num_errors = 0
    num_bit_errors = 0
    num_frames = 0
    total_decode_time = 0.0
    total_iters = 0.0

    while num_frames < max_frames and num_errors < min_errors:
      if crc_length > 0:
        info_bits = rng.integers(0, 2, K_info)
        payload = crc_encode(info_bits, crc_length)
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_indices] = payload
      else:
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_indices] = rng.integers(0, 2, len(info_indices))

      x = polar_encode(u_sent)
      y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
      llr = compute_llr(y, sigma)

      t0 = time.perf_counter()
      u_hat, aux = decoder(llr)
      total_decode_time += time.perf_counter() - t0

      if aux is not None and decoder_type == "bp":
        total_iters += aux

      if crc_length > 0:
        frame_err = not np.array_equal(u_hat[info_indices], u_sent[info_indices])
        bit_err = np.sum(u_hat[info_indices[:K_info]] != u_sent[info_indices[:K_info]])
      else:
        frame_err = not np.array_equal(u_hat[info_indices], u_sent[info_indices])
        bit_err = np.sum(u_hat[info_indices] != u_sent[info_indices])

      num_frames += 1
      if frame_err:
        num_errors += 1
      num_bit_errors += bit_err

    bler = num_errors / num_frames
    ber = num_bit_errors / (num_frames * K_info) if K_info > 0 else 0.0
    avg_time = total_decode_time / num_frames
    avg_iters = (total_iters / num_frames) if decoder_type == "bp" else None

    result = {
      "eb_n0_db": eb_n0_db,
      "bler": bler,
      "ber": ber,
      "num_errors": num_errors,
      "num_frames": num_frames,
      "avg_decode_time": avg_time,
      "avg_iters": avg_iters,
    }
    results.append(result)

    if verbose:
      msg = (
        f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
        f"| Errors={num_errors} | Frames={num_frames} "
        f"| AvgTime={avg_time * 1000:.2f}ms"
      )
      if avg_iters is not None:
        msg += f" | AvgIter={avg_iters:.1f}"
      print(msg)

  return results
