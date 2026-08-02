"""
蒙特卡洛仿真主循环
"""
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode
from encoder import polar_encode


def run_unit_tests():
  """运行各模块单元测试。"""
  print("=" * 50)
  print("运行单元测试...")
  print("=" * 50)

  # 编码器校验: u=[1,0,1,1] -> x=[1,0,1,1] (G_N = B_N F^{⊗n})
  u = np.array([1, 0, 1, 1])
  x = polar_encode(u)
  assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"
  u2 = np.array([1, 1, 0, 0])
  x2 = polar_encode(u2)
  assert np.array_equal(x2, [0, 0, 1, 0]), f"编码器错误: {x2}"
  print("[PASS] 编码器校验")

  # GA 构造校验
  info8, frozen8, _ = ga_construction(8, 4, 2.5)
  print(f"[INFO] N=8 info_indices: {info8}, frozen: {frozen8}")

  # SC 译码校验（无损）
  N, K = 64, 32
  info_idx, _, _ = ga_construction(N, K, 2.5)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0
  rng = np.random.default_rng(0)
  rate = K / N
  sigma = eb_n0_to_sigma(10.0, rate)
  errors = 0
  for _ in range(100):
    info_bits = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info_bits
    x = polar_encode(u)
    s = bpsk_modulate(x)
    y = awgn_channel(s, sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat = sc_decode(llr, frozen_bits)
    if not np.array_equal(u_hat[info_idx], info_bits):
      errors += 1
  assert errors == 0, f"SC 译码在 Eb/N0=10dB 有 {errors} 帧错误"
  print("[PASS] SC 译码无损校验 (N=64, 100帧)")

  # 路径度量校验：L=1 SCL 等价于 SC
  scl = SCLDecoder(N, frozen_bits, list_size=1)
  errors_scl = 0
  for _ in range(50):
    info_bits = rng.integers(0, 2, K)
    u = np.zeros(N, dtype=int)
    u[info_idx] = info_bits
    x = polar_encode(u)
    s = bpsk_modulate(x)
    y = awgn_channel(s, sigma, rng)
    llr = compute_llr(y, sigma)
    u_hat, _ = scl.decode(llr)
    if not np.array_equal(u_hat[info_idx], info_bits):
      errors_scl += 1
  assert errors_scl == 0, f"L=1 SCL 有 {errors_scl} 帧错误"
  print("[PASS] L=1 SCL 等价 SC 校验")
  print("=" * 50)
  print("所有单元测试通过！\n")


def run_simulation(
    N, K, eb_n0_db_list, decoder,
    decoder_type="sc",
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    info_indices=None,
    verbose=True,
    seed=42,
):
  """
  蒙特卡洛仿真。
  """
  rng = np.random.default_rng(seed)
  rate = K / N
  K_info = K - crc_length
  results = []

  if info_indices is None:
    info_idx, _, _ = ga_construction(N, K, 2.5)
  else:
    info_idx = np.asarray(info_indices)

  for eb_n0_db in eb_n0_db_list:
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    num_errors = 0
    num_bit_errors = 0
    num_frames = 0
    total_decode_time = 0.0
    total_iters = 0

    while num_frames < max_frames and num_errors < min_errors:
      if crc_length > 0:
        info_bits = rng.integers(0, 2, K_info)
        payload = crc_encode(info_bits, crc_length)
      else:
        payload = rng.integers(0, 2, K)

      u = np.zeros(N, dtype=int)
      u[info_idx] = payload

      x = polar_encode(u)
      s = bpsk_modulate(x)
      y = awgn_channel(s, sigma, rng)
      llr = compute_llr(y, sigma)

      t0 = time.perf_counter()
      u_hat, aux = decoder(llr)
      total_decode_time += time.perf_counter() - t0

      if aux is not None and decoder_type == "bp":
        total_iters += aux

      num_frames += 1
      if crc_length > 0:
        check_bits = u_hat[info_idx]
        frame_ok = np.array_equal(check_bits[:K_info], info_bits)
      else:
        frame_ok = np.array_equal(u_hat[info_idx], payload)

      if not frame_ok:
        num_errors += 1
        if crc_length > 0:
          num_bit_errors += np.sum(u_hat[info_idx][:K_info] != info_bits)
        else:
          num_bit_errors += np.sum(u_hat[info_idx] != payload)

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
