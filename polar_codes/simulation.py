"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from construction import ga_construction_bh as ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_scl import crc_encode


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
    info_idx=None,
    design_ebn0=2.5,
):
    """
    蒙特卡洛仿真。

    返回每个 Eb/N0 点的结果字典列表。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length

    if info_idx is None:
        info_idx, _, _ = ga_construction(N, K, design_ebn0)

    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            if crc_length > 0:
                info_bits = rng.integers(0, 2, k_info)
                payload = crc_encode(info_bits, crc_length)
                u[info_idx] = payload
            else:
                u[info_idx] = rng.integers(0, 2, K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            out = decoder(llr)
            t1 = time.perf_counter()
            total_decode_time += t1 - t0

            if isinstance(out, tuple):
                u_hat, aux = out
            else:
                u_hat, aux = out, None

            if aux is not None and decoder_type == "bp":
                total_iters += aux

            if crc_length > 0:
                check_bits = u[info_idx][:k_info]
                est_bits = u_hat[info_idx][:k_info]
            else:
                check_bits = u[info_idx]
                est_bits = u_hat[info_idx]

            bit_err = np.sum(check_bits != est_bits)
            num_bit_errors += bit_err
            if bit_err > 0:
                num_errors += 1
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
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
