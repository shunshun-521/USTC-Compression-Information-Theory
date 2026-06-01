"""
蒙特卡洛仿真主循环
"""
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_scl import crc_encode
from encoder import polar_encode


def run_simulation(
  N,
  K,
  eb_n0_db_list,
  decoder,
  decoder_type="sc",
  max_frames=100000,
  min_errors=100,
  crc_length=0,
  design_eb_n0_db=2.5,
  verbose=True,
  seed=42,
):
  """
  蒙特卡洛仿真。
  """
  rng = np.random.default_rng(seed)
  rate = K / N
  info_idx, _, _ = ga_construction(N, K, design_eb_n0_db, rate)
  frozen_bits = np.ones(N, dtype=int)
  frozen_bits[info_idx] = 0

  if crc_length > 0:
    K_info = K - crc_length
    info_payload_idx = info_idx[:K_info]
    crc_idx = info_idx[K_info:]
  else:
    K_info = K
    info_payload_idx = info_idx

  results = []

  for eb_n0_db in eb_n0_db_list:
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    num_errors = 0
    num_bit_errors = 0
    num_frames = 0
    total_decode_time = 0.0
    total_iters = 0

    while num_frames < max_frames and num_errors < min_errors:
      u = np.zeros(N, dtype=int)

      if crc_length > 0:
        payload = rng.integers(0, 2, size=K_info)
        with_crc = crc_encode(payload, crc_length)
        u[info_idx] = with_crc
      else:
        payload = rng.integers(0, 2, size=K_info)
        u[info_payload_idx] = payload

      x = polar_encode(u)
      y = awgn_channel(bpsk_modulate(x), sigma, rng)
      llr = compute_llr(y, sigma)

      t0 = time.perf_counter()
      decode_out = decoder(llr)
      t1 = time.perf_counter()

      if isinstance(decode_out, tuple):
        u_hat, aux = decode_out
      else:
        u_hat, aux = decode_out, None

      total_decode_time += t1 - t0
      if aux is not None and decoder_type == "bp":
        total_iters += aux

      if crc_length > 0:
        err = not np.array_equal(u_hat[info_payload_idx], payload)
        bit_err = np.sum(u_hat[info_payload_idx] != payload)
      else:
        err = not np.array_equal(u_hat[info_payload_idx], payload)
        bit_err = np.sum(u_hat[info_payload_idx] != payload)

      num_frames += 1
      if err:
        num_errors += 1
      num_bit_errors += bit_err

    bler = num_errors / num_frames if num_frames else 1.0
    ber = num_bit_errors / (num_frames * K_info) if num_frames else 1.0
    avg_time = total_decode_time / num_frames if num_frames else 0.0
    avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames else None

    result = {
      "eb_n0_db": float(eb_n0_db),
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
