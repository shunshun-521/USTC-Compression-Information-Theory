"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode, bit_reversal_permutation


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
    rate=None,
    info_idx=None,
    frozen_bits=None,
):
    """蒙特卡洛仿真。"""
    rng = np.random.default_rng(seed)
    if rate is None:
        rate = K / N
    br = bit_reversal_permutation(N)

    if info_idx is None or frozen_bits is None:
        from construction import ga_construction

        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=bool)
        frozen_bits[info_idx] = False

    info_idx = np.asarray(info_idx, dtype=int)
    K_info = len(info_idx) - crc_length
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
                from decoder_scl import crc_encode

                info_bits = rng.integers(0, 2, K_info)
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = rng.integers(0, 2, len(info_idx))

            u[info_idx] = payload

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x, rate), sigma, rng)
            llr = compute_llr(y, sigma, rate)
            llr_dec = llr[br]

            t0 = time.perf_counter()
            out = decoder(llr_dec)
            if isinstance(out, tuple):
                u_hat, aux = out
            else:
                u_hat, aux = out, None
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                from decoder_scl import crc_check

                frame_err = not crc_check(u_hat[info_idx], crc_length)
                bit_err = np.sum(u_hat[info_idx][:K_info] != u[info_idx][:K_info])
            else:
                frame_err = not np.array_equal(u_hat[info_idx], u[info_idx])
                bit_err = np.sum(u_hat[info_idx] != u[info_idx])

            num_frames += 1
            num_bit_errors += bit_err
            if frame_err:
                num_errors += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if K_info else 0.0
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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters else "")
            )

    return results
