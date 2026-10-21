"""
蒙特卡洛仿真主循环
"""
import os
import time
import numpy as np

from channel import (
    bpsk_modulate,
    awgn_channel,
    compute_llr,
    eb_n0_to_sigma,
)
from encoder import polar_encode
from decoder_scl import crc_encode


def _env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


def _env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type="sc",
    max_frames=None,
    min_errors=None,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
    frozen_bits=None,
):
    """
    蒙特卡洛仿真。
    环境变量 POLAR_MAX_FRAMES / POLAR_MIN_ERRORS 可覆盖默认帧数设置。
    """
    if max_frames is None:
        max_frames = _env_int("POLAR_MAX_FRAMES", 100000)
    if min_errors is None:
        min_errors = _env_int("POLAR_MIN_ERRORS", 100)

    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length
    results = []

    if frozen_bits is None:
        raise ValueError("frozen_bits must be provided")
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    if info_indices is None:
        info_indices = np.where(frozen_bits == 0)[0]

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info = rng.integers(0, 2, size=K_info, dtype=int)
                payload = crc_encode(info, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = payload
            else:
                u = np.zeros(N, dtype=int)
                payload = rng.integers(0, 2, size=K, dtype=int)
                u[info_indices] = payload
                info = payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if aux is not None and decoder_type == "bp":
                total_iters += aux

            if crc_length > 0:
                frame_ok = np.array_equal(u_hat[info_indices], u[info_indices])
            else:
                frame_ok = np.array_equal(u_hat[info_indices], u[info_indices])

            if not frame_ok:
                num_errors += 1
                if crc_length > 0:
                    num_bit_errors += np.count_nonzero(
                        u_hat[info_indices[:K_info]] != u[info_indices[:K_info]]
                    )
                else:
                    num_bit_errors += np.count_nonzero(
                        u_hat[info_indices] != u[info_indices]
                    )

            num_frames += 1

        bler = num_errors / num_frames if num_frames else 1.0
        ber_denom = num_frames * (K_info if crc_length > 0 else len(info_indices))
        ber = num_bit_errors / ber_denom if num_frames and ber_denom else 1.0
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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
