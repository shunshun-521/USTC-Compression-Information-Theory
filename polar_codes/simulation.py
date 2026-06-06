"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import (
    bpsk_modulate,
    awgn_channel,
    compute_llr,
    eb_n0_to_es,
    prepare_decoder_llr,
    prepare_frozen_bits_decoder,
    map_decoder_bits_to_natural,
)
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
    verbose=True,
    seed=42,
    info_indices=None,
    frozen_bits=None,
    design_eb_n0_db=2.5,
):
    """蒙特卡洛仿真"""
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length

    if info_indices is None or frozen_bits is None:
        info_idx, _, _ = ga_construction(N, K, design_eb_n0_db, rate=rate)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0
        info_indices = info_idx
    else:
        info_indices = np.asarray(info_indices)
        frozen_bits = np.asarray(frozen_bits)

    frozen_dec = prepare_frozen_bits_decoder(frozen_bits, N)
    results = []

    for eb_n0_db in eb_n0_db_list:
        es = eb_n0_to_es(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_raw = rng.integers(0, 2, size=K_info)
                payload = crc_encode(info_raw, crc_length)
            else:
                payload = rng.integers(0, 2, size=K_info)

            u = np.zeros(N, dtype=int)
            u[info_indices] = payload

            x = polar_encode(u)
            s = bpsk_modulate(x, es=es)
            y = awgn_channel(s, es=es, rng=rng)
            llr = prepare_decoder_llr(compute_llr(y, es=es), N)

            t0 = time.perf_counter()
            u_hat_dec, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == 'bp' and aux is not None:
                total_iters += aux

            u_hat = map_decoder_bits_to_natural(u_hat_dec, N)

            frame_err = not np.array_equal(payload, u_hat[info_indices])
            if frame_err:
                num_errors += 1
                num_bit_errors += np.sum(payload != u_hat[info_indices])

            num_frames += 1

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames else 0.0
        avg_time = total_decode_time / num_frames if num_frames else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == 'bp' else None

        result = {
            'eb_n0_db': float(eb_n0_db),
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
