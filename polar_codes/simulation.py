"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


def _sim_limits(max_frames, min_errors):
    if os.environ.get("POLAR_FAST_SIM", "").strip() in ("1", "true", "yes"):
        return min(max_frames, 2000), min(min_errors, 30)
    return max_frames, min_errors


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
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
    info_indices: 信息位在 u 向量中的索引；若为 None 则使用前 K 个索引（不推荐）。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    max_frames, min_errors = _sim_limits(max_frames, min_errors)
    K_info = K - crc_length

    if info_indices is None:
        info_indices = np.arange(K, dtype=np.int64)
    else:
        info_indices = np.asarray(info_indices, dtype=np.int64)

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
                info = rng.integers(0, 2, size=K_info, dtype=np.int8)
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

            if aux is not None and decoder_type == "bp":
                total_iters += float(aux)

            rcv_payload = u_hat[info_indices]
            if crc_length > 0:
                sent_cmp = info
                rcv_cmp = rcv_payload[:K_info]
            else:
                sent_cmp = payload
                rcv_cmp = rcv_payload

            frame_err = not np.array_equal(sent_cmp, rcv_cmp)
            if frame_err:
                num_errors += 1
            num_bit_errors += np.count_nonzero(sent_cmp != rcv_cmp)
            num_frames += 1

        bler = num_errors / num_frames if num_frames else 1.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames and K_info else 1.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames else None

        result = {
            "eb_n0_db": float(eb_n0_db),
            "bler": float(bler),
            "ber": float(ber),
            "num_errors": int(num_errors),
            "num_frames": int(num_frames),
            "avg_decode_time": float(avg_time),
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
