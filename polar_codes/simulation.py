"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, channel_params
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
    """蒙特卡洛仿真"""
    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length
    results = []

    for eb_n0_db in eb_n0_db_list:
        es, no, sigma = channel_params(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            if info_indices is None:
                info_idx = np.where(np.arange(N) < K)[0]
            else:
                info_idx = info_indices

            if crc_length > 0:
                payload = rng.integers(0, 2, k_info)
                coded_info = crc_encode(payload, crc_length)
                u[info_idx] = coded_info
            else:
                u[info_idx] = rng.integers(0, 2, k_info)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x, es), sigma, rng)
            llr = compute_llr(y, Es=es, No=no)

            t0 = time.perf_counter()
            out = decoder(llr)
            dt = time.perf_counter() - t0
            total_decode_time += dt

            if isinstance(out, tuple):
                u_hat, aux = out
                if decoder_type == "bp" and aux is not None:
                    total_iters += aux
            else:
                u_hat = out

            if crc_length > 0:
                sent_payload = u[info_idx][:k_info]
                recv_payload = u_hat[info_idx][:k_info]
                bit_err = np.sum(sent_payload != recv_payload)
            else:
                bit_err = np.sum(u[info_idx] != u_hat[info_idx])

            frame_err = bit_err > 0 if crc_length > 0 else not np.array_equal(u[info_idx], u_hat[info_idx])
            if frame_err:
                num_errors += 1
            num_bit_errors += bit_err
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * k_info)
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
