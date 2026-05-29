"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from encoder import polar_encode
from decoder_scl import crc_encode


def run_simulation(
    N,
    K,
    eb_n0_db_list,
    decoder,
    decoder_type='sc',
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    design_eb_n0_db=2.5,
    verbose=True,
    seed=42,
):
    """
    蒙特卡洛仿真。

    返回每个 Eb/N0 点的结果字典列表。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    info_idx, _, _ = ga_construction(N, K, design_eb_n0_db, rate=rate)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    K_info = K - crc_length
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
            msg = rng.integers(0, 2, K_info)
            if crc_length > 0:
                payload = crc_encode(msg, crc_length)
                u[info_idx] = payload
            else:
                u[info_idx] = msg

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            out = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if isinstance(out, tuple):
                u_hat, aux = out
                if decoder_type == 'bp' and aux is not None:
                    total_iters += aux
            else:
                u_hat = out

            u_hat = np.asarray(u_hat, dtype=int)
            if crc_length > 0:
                check_bits = u_hat[info_idx]
                frame_ok = crc_check_bits(check_bits, crc_length) and np.array_equal(
                    u_hat[info_idx][:K_info], msg
                )
            else:
                frame_ok = np.array_equal(u_hat[info_idx], msg)

            if not frame_ok:
                num_errors += 1
                num_bit_errors += np.sum(u_hat[info_idx][:K_info] != msg)
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if K_info > 0 else 0.0
        avg_time = total_decode_time / num_frames
        avg_iters = (total_iters / num_frames) if decoder_type == 'bp' else None

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
            msg = (
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
            )
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results


def crc_check_bits(bits, crc_length):
    from decoder_scl import crc_check
    return crc_check(bits, crc_length)
