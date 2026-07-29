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
    print("Running unit tests...")

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"
    print("  [PASS] Encoder test")

    N, K = 64, 32
    design_eb_n0 = 2.5
    info_idx, _, _ = ga_construction(N, K, design_eb_n0)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(123)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    errors = 0
    for _ in range(100):
        info_bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], info_bits):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors}/100 错误"
    print("  [PASS] SC decode test (100 frames, Eb/N0=10dB)")

    scl = SCLDecoder(N, frozen_bits, list_size=1)
    errors_scl = 0
    for _ in range(50):
        info_bits = rng.integers(0, 2, K)
        u = np.zeros(N, dtype=int)
        u[info_idx] = info_bits
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat_sc = sc_decode(llr, frozen_bits)
        u_hat_scl, _ = scl.decode(llr)
        if not np.array_equal(u_hat_sc, u_hat_scl):
            errors_scl += 1
    assert errors_scl == 0, f"L=1 SCL 与 SC 不一致: {errors_scl}/50"
    print("  [PASS] SCL L=1 equals SC test")

    print("All unit tests passed.\n")


def run_simulation(
    N, K, eb_n0_db_list, decoder,
    decoder_type='sc',
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
    design_eb_n0_db=2.5,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N

    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)

    info_indices = np.asarray(info_indices)
    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            if crc_length > 0:
                K_info = K - crc_length
                info_bits_raw = rng.integers(0, 2, K_info)
                info_with_crc = crc_encode(info_bits_raw, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices] = info_with_crc
            else:
                info_bits_raw = rng.integers(0, 2, K)
                u = np.zeros(N, dtype=int)
                u[info_indices] = info_bits_raw

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == 'bp' and aux is not None:
                total_iters += aux

            if crc_length > 0:
                decoded_info = u_hat[info_indices][:K_info]
                if not np.array_equal(decoded_info, info_bits_raw):
                    num_errors += 1
                    num_bit_errors += np.sum(decoded_info != info_bits_raw)
            else:
                decoded_info = u_hat[info_indices]
                if not np.array_equal(decoded_info, info_bits_raw):
                    num_errors += 1
                    num_bit_errors += np.sum(decoded_info != info_bits_raw)

            num_frames += 1

        K_info = K - crc_length if crc_length > 0 else K
        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if num_frames > 0 else 0.0
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
            print(
                f"  Eb/N0={eb_n0_db:.2f}dB | BLER={bler:.4e} | BER={ber:.4e} "
                f"| Errors={num_errors} | Frames={num_frames} "
                f"| AvgTime={avg_time * 1000:.2f}ms"
                + (f" | AvgIter={avg_iters:.1f}" if avg_iters is not None else "")
            )

    return results
