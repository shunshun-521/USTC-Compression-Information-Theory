"""
蒙特卡洛仿真主循环
"""
import os
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
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
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            if info_indices is None:
                raise ValueError("info_indices required")
            if crc_length > 0:
                info_payload = rng.integers(0, 2, size=k_info)
                info_with_crc = crc_encode(info_payload, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = info_with_crc
            else:
                u = np.zeros(N, dtype=int)
                bits = rng.integers(0, 2, size=K)
                u[info_indices] = bits
                info_payload = bits

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                decoded_info = u_hat[info_indices]
                frame_ok = crc_check_payload(decoded_info, info_payload, crc_length)
            else:
                frame_ok = np.array_equal(u_hat[info_indices], info_payload)

            if not frame_ok:
                num_errors += 1
            num_bit_errors += np.count_nonzero(u_hat[info_indices][:k_info] != info_payload[:k_info])
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


def crc_check_payload(decoded, original_payload, crc_length):
    from decoder_scl import crc_check

    if len(decoded) != len(original_payload) + crc_length:
        return False
    return crc_check(decoded, crc_length)


def apply_quick_env(default_max_frames, default_min_errors, default_eb_list):
    """读取 POLAR_QUICK / POLAR_MAX_FRAMES / POLAR_MIN_ERRORS 环境变量"""
    if os.environ.get("POLAR_QUICK", "").lower() in ("1", "true", "yes"):
        default_max_frames = min(default_max_frames, 2000)
        default_min_errors = min(default_min_errors, 20)
        if len(default_eb_list) > 4:
            default_eb_list = default_eb_list[len(default_eb_list) // 2 : len(default_eb_list) // 2 + 3]
    if "POLAR_MAX_FRAMES" in os.environ:
        default_max_frames = int(os.environ["POLAR_MAX_FRAMES"])
    if "POLAR_MIN_ERRORS" in os.environ:
        default_min_errors = int(os.environ["POLAR_MIN_ERRORS"])
    return default_max_frames, default_min_errors, default_eb_list
