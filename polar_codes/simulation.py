"""
蒙特卡洛仿真主循环
"""
import os
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
    verbose=True,
    seed=42,
    info_indices=None,
    frozen_bits=None,
    design_eb_n0_db=2.5,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    if info_indices is None or frozen_bits is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_indices] = 0

    info_indices = np.asarray(info_indices)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    k_payload = K - crc_length

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
                info = rng.integers(0, 2, size=k_payload, dtype=np.int8)
                payload = crc_encode(info, crc_length)
            else:
                payload = rng.integers(0, 2, size=K, dtype=np.int8)

            u = np.zeros(N, dtype=np.int8)
            u[info_indices] = payload

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng=rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            num_frames += 1
            if crc_length > 0:
                sent_payload = payload[:k_payload]
                recv_payload = u_hat[info_indices][:k_payload]
            else:
                sent_payload = payload
                recv_payload = u_hat[info_indices]

            if not np.array_equal(recv_payload, sent_payload):
                num_errors += 1
                num_bit_errors += np.count_nonzero(recv_payload != sent_payload)

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_payload) if k_payload else 0.0
        avg_time = total_decode_time / num_frames
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" else None

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


def env_int(name, default):
    v = os.environ.get(name)
    return int(v) if v is not None else default


def env_float(name, default):
    v = os.environ.get(name)
    return float(v) if v is not None else default


def simulation_defaults(max_frames, min_errors, eb_range):
    """允许通过环境变量缩短自动化运行时间"""
    max_f = env_int("POLAR_MAX_FRAMES", max_frames)
    min_e = env_int("POLAR_MIN_ERRORS", min_errors)
    eb_min = env_float("POLAR_EB_MIN", eb_range[0])
    eb_max = env_float("POLAR_EB_MAX", eb_range[-1] if len(eb_range) else eb_range[0])
    eb_step = env_float("POLAR_EB_STEP", eb_range[1] - eb_range[0] if len(eb_range) > 1 else 0.25)
    eb_list = np.arange(eb_min, eb_max + 0.5 * eb_step, eb_step)
    return max_f, min_e, eb_list
