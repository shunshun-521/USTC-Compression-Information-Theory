"""
蒙特卡洛仿真主循环
"""
import numpy as np
import time
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode
from decoder_scl import crc_encode


def run_simulation(
    N, K, eb_n0_db_list, decoder,
    decoder_type='sc',
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    info_indices=None,
    verbose=True,
    seed=42
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    K_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            # 生成信息比特
            info_bits = rng.integers(0, 2, size=K_info)

            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
            else:
                payload = info_bits

            # 构造源序列 u
            u = np.zeros(N, dtype=int)
            if info_indices is None:
                info_indices_local = np.where(np.arange(N) < K)[0]  # fallback
            else:
                info_indices_local = info_indices
            u[info_indices_local] = payload

            # 编码
            x = polar_encode(u)

            # 信道
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            # 译码
            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            t1 = time.perf_counter()
            total_decode_time += (t1 - t0)

            if decoder_type == 'bp' and aux is not None:
                total_iters += aux

            # 比较信息比特
            decoded_info = u_hat[info_indices_local][:K_info]
            frame_error = not np.array_equal(decoded_info, info_bits)
            if frame_error:
                num_errors += 1
            num_bit_errors += np.sum(decoded_info != info_bits)

            num_frames += 1

        bler = num_errors / num_frames if num_frames > 0 else 0
        ber = num_bit_errors / (num_frames * K_info) if num_frames > 0 else 0
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0
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
            print(f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                  f"| Errors={num_errors} | Frames={num_frames} "
                  f"| AvgTime={avg_time * 1000:.2f}ms"
                  + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else ""))

    return results
