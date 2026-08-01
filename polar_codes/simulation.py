"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from channel import (
    bpsk_modulate,
    awgn_channel,
    compute_llr,
    eb_n0_to_sigma,
    prepare_decoder_llr,
)
from encoder import polar_encode
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
    info_indices=None,
):
    """蒙特卡洛仿真。"""
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []
    k_info = K - crc_length

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
                payload = rng.integers(0, 2, k_info)
                coded = crc_encode(payload, crc_length)
                if info_indices is not None:
                    u[info_indices[:K]] = coded
                else:
                    u[:K] = coded
                ref_info = payload
                ref_slice = slice(0, k_info) if info_indices is None else info_indices[:k_info]
            else:
                payload = rng.integers(0, 2, K)
                if info_indices is not None:
                    u[info_indices] = payload
                    ref_slice = info_indices
                else:
                    u[:K] = payload
                    ref_slice = slice(0, K)
                ref_info = payload

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = prepare_decoder_llr(compute_llr(y, sigma), N)

            t0 = time.perf_counter()
            dec_out = decoder(llr)
            t1 = time.perf_counter()
            if isinstance(dec_out, tuple):
                u_hat, aux = dec_out
            else:
                u_hat, aux = dec_out, None
            total_decode_time += t1 - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if isinstance(ref_slice, slice):
                hat_info = u_hat[ref_slice]
            else:
                hat_info = u_hat[ref_slice]

            frame_err = not np.array_equal(hat_info, ref_info)
            num_errors += int(frame_err)
            num_bit_errors += np.sum(hat_info != ref_info)
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info) if k_info > 0 else 0.0
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
