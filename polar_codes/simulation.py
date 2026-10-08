"""
蒙特卡洛仿真主循环
"""
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
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                info_bits = rng.integers(0, 2, k_info)
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = rng.integers(0, 2, K)

            u = np.zeros(N, dtype=np.int8)
            u[~np.asarray(decoder.frozen_mask, dtype=bool)] = payload

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            out = decoder(llr)
            t1 = time.perf_counter()
            if isinstance(out, tuple):
                u_hat, aux = out[0], out[1]
            else:
                u_hat, aux = out, None
            total_decode_time += t1 - t0
            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                sent_info = info_bits
                recv_info = u_hat[~decoder.frozen_mask][:k_info]
            else:
                sent_info = payload
                recv_info = u_hat[~decoder.frozen_mask]

            frame_err = not np.array_equal(recv_info, sent_info)
            num_bit_errors += int(np.sum(recv_info != sent_info))
            num_errors += int(frame_err)
            num_frames += 1

        bler = num_errors / num_frames if num_frames else 0.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames and k_info else 0.0
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


class DecoderWrapper:
    """包装译码器，携带 frozen_mask 供仿真填充信息位。"""

    def __init__(self, frozen_bits, decode_fn):
        self.frozen_mask = np.asarray(frozen_bits, dtype=bool)
        self._decode_fn = decode_fn

    def __call__(self, llr):
        return self._decode_fn(llr)
