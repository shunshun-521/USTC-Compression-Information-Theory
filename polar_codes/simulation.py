"""
蒙特卡洛仿真主循环
"""
import os
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from encoder import bit_reversal_permutation, polar_encode


def run_unit_tests():
    """运行各模块数值正确性校验"""
    from decoder_sc import sc_decode
    from decoder_scl import SCLDecoder

    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rev = bit_reversal_permutation(N)
    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(10.0, rate)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr[rev], frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC 译码在 Eb/N0=10dB 下有 {errors} 帧错误"

    # SCL L=1 应等价于 SC
    rng = np.random.default_rng(1)
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_sent)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr[rev], frozen_bits)
        u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr[rev])
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("所有单元测试通过。")


def run_simulation(
    N, K, eb_n0_db_list, decoder,
    decoder_type="sc",
    max_frames=100000,
    min_errors=100,
    crc_length=0,
    verbose=True,
    seed=42,
    info_indices=None,
    frozen_bits=None,
    design_eb_n0_db=2.5,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N

    if info_indices is None or frozen_bits is None:
        info_indices, _, _ = ga_construction(N, K, design_eb_n0_db)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_indices] = 0

    info_indices = np.asarray(info_indices, dtype=int)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    rev = bit_reversal_permutation(N)

    K_info = K - crc_length
    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u_sent = np.zeros(N, dtype=int)

            if crc_length > 0:
                from decoder_scl import crc_encode

                info_payload = rng.integers(0, 2, size=K_info)
                payload_with_crc = crc_encode(info_payload, crc_length)
                u_sent[info_indices] = payload_with_crc
            else:
                u_sent[info_indices] = rng.integers(0, 2, size=K)

            x = polar_encode(u_sent)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)
            llr_dec = llr[rev]

            t0 = time.perf_counter()
            decode_out = decoder(llr_dec)
            total_decode_time += time.perf_counter() - t0

            if isinstance(decode_out, tuple):
                u_hat, aux = decode_out
            else:
                u_hat, aux = decode_out, None

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                frame_err = not np.array_equal(
                    u_hat[info_indices][:K_info],
                    u_sent[info_indices][:K_info],
                )
            else:
                frame_err = not np.array_equal(u_hat[info_indices], u_sent[info_indices])

            if frame_err:
                num_errors += 1
                if crc_length > 0:
                    num_bit_errors += np.sum(
                        u_hat[info_indices][:K_info] != u_sent[info_indices][:K_info]
                    )
                else:
                    num_bit_errors += np.sum(u_hat[info_indices] != u_sent[info_indices])

            num_frames += 1

        bler = num_errors / num_frames if num_frames > 0 else 0.0
        ber = num_bit_errors / (num_frames * K_info) if num_frames > 0 else 0.0
        avg_time = total_decode_time / num_frames if num_frames > 0 else 0.0
        avg_iters = (total_iters / num_frames) if decoder_type == "bp" and num_frames > 0 else None

        result = {
            "eb_n0_db": float(eb_n0_db),
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
