"""
蒙特卡洛仿真主循环
"""
import time
import os

import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from encoder import polar_encode


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type='sc',
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
    K_info = K - crc_length

    if info_indices is None or frozen_bits is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db, rate)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_indices] = 0

    info_indices = np.asarray(info_indices, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=int)

    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                from decoder_scl import crc_encode

                info_payload = rng.integers(0, 2, size=K_info)
                payload_with_crc = crc_encode(info_payload, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = payload_with_crc
            else:
                u = np.zeros(N, dtype=int)
                u[info_indices] = rng.integers(0, 2, size=K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == 'bp':
                total_iters += aux

            if crc_length > 0:
                if not np.array_equal(u_hat[info_indices], u[info_indices]):
                    num_errors += 1
                    num_bit_errors += np.sum(u_hat[info_indices] != u[info_indices])
            else:
                if not np.array_equal(u_hat[info_indices], u[info_indices]):
                    num_errors += 1
                    num_bit_errors += np.sum(u_hat[info_indices] != u[info_indices])

            num_frames += 1

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames else 0.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == 'bp' and num_frames else None

        result = {
            'eb_n0_db': eb_n0_db,
            'bler': bler,
            'ber': ber,
            'num_errors': num_errors,
            'num_frames': num_frames,
            'avg_decode_time': avg_time,
            'avg_iters': avg_iters,
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


def fast_sim_settings():
    """快速仿真参数（用于测试）。"""
    if os.environ.get('POLAR_FAST_SIM', '0') == '1':
        return {
            'max_frames': 2000,
            'min_errors': 20,
            'eb_n0_range_sc': np.arange(1.0, 4.0, 0.5),
            'eb_n0_range_other': np.arange(1.5, 4.0, 0.5),
            'n_list_exp1': [256, 512],
        }
    return None
