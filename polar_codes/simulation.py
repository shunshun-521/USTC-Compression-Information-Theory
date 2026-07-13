"""
蒙特卡洛仿真主循环
"""
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode
from encoder import polar_encode


def run_unit_tests():
    """运行各模块单元测试。"""
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(123)
    sc_errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits.astype(bool))
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            sc_errors += 1
    assert sc_errors == 0, f"SC 译码校验失败: {sc_errors}/100 错误"

    scl = SCLDecoder(N, frozen_bits.astype(bool), list_size=1)
    scl_errors = 0
    rng2 = np.random.default_rng(123)
    for _ in range(50):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng2.integers(0, 2, K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng2)
        llr = compute_llr(y, sigma)
        u_hat, _ = scl.decode(llr)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            scl_errors += 1
    assert scl_errors == 0, f"SCL L=1 应等价于 SC: {scl_errors}/50 错误"

    print("单元测试全部通过。")


def run_simulation(
    N, K, eb_n0_db_list, decoder,
    decoder_type='sc',
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    verbose=True,
    seed=42,
    info_idx=None,
    frozen_bits=None,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    results = []

    if info_idx is None or frozen_bits is None:
        info_idx, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_idx] = 0

    k_info = K - crc_length

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u_info = rng.integers(0, 2, k_info)

            u = np.zeros(N, dtype=int)
            if crc_length > 0:
                payload = crc_encode(u_info, crc_length)
                u[info_idx] = payload
            else:
                u[info_idx] = u_info

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            decode_out = decoder(llr)
            t1 = time.perf_counter()

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
            else:
                u_hat, aux = decode_out, None

            total_decode_time += (t1 - t0)
            if aux is not None and decoder_type == 'bp':
                total_iters += aux

            if crc_length > 0:
                compare_info = u_info
                decoded_info = u_hat[info_idx][:k_info]
            else:
                compare_info = u_info
                decoded_info = u_hat[info_idx]

            frame_error = not np.array_equal(decoded_info, compare_info)
            if frame_error:
                num_errors += 1
            num_bit_errors += np.sum(decoded_info != compare_info)
            num_frames += 1

        bler = num_errors / num_frames if num_frames > 0 else 0.0
        ber = num_bit_errors / (num_frames * k_info) if num_frames > 0 else 0.0
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == 'bp' and num_frames > 0 else None

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
            msg = (f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                   f"| Errors={num_errors} | Frames={num_frames} "
                   f"| AvgTime={avg_time * 1000:.2f}ms")
            if avg_iters is not None:
                msg += f" | AvgIter={avg_iters:.1f}"
            print(msg)

    return results
