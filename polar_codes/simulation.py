"""
蒙特卡洛仿真主循环
"""
import time

import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from encoder import polar_encode, polar_generator_matrix
from decoder_sc import sc_decode, sc_decode_recursive
from decoder_scl import SCLDecoder, crc_encode


def run_unit_tests():
    """运行各模块数值正确性校验。"""
    # 编码器校验：与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = polar_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} != {x_ref}"

    # SC 译码校验（无损，使用非递归实现）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    rng = np.random.default_rng(0)
    rate = K / N
    sigma = eb_n0_to_sigma(15.0, rate)
    errors = 0
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        u_sent[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u_sent)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u_hat[info_idx], u_sent[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损译码失败，错误帧数={errors}"

    # 递归与非递归一致性（小码长）
    N_small = 16
    info_s, _, _ = ga_construction(N_small, 8, 2.5)
    fb_s = np.ones(N_small, dtype=int)
    fb_s[info_s] = 0
    llr_s = rng.normal(0, 2, N_small)
    assert np.array_equal(
        sc_decode(llr_s, fb_s), sc_decode_recursive(llr_s, fb_s)
    ), "SC 递归与非递归结果不一致"

    # 路径度量校验：L=1 SCL 等价于 SC
    llr_test = rng.normal(0, 1, N)
    u_sc = sc_decode(llr_test, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
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
):
    """蒙特卡洛仿真。"""
    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length

    if info_indices is None:
        info_indices, _, _ = ga_construction(N, K, 2.5)

    results = []

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0.0

        while num_frames < max_frames and num_errors < min_errors:
            u_sent = np.zeros(N, dtype=int)
            info_bits = rng.integers(0, 2, k_info)
            if crc_length > 0:
                payload = crc_encode(info_bits, crc_length)
                u_sent[info_indices] = payload
            else:
                u_sent[info_indices] = info_bits

            x = polar_encode(u_sent)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None:
                total_iters += aux

            if not np.array_equal(u_hat[info_indices][:k_info], info_bits):
                num_errors += 1
                num_bit_errors += np.sum(u_hat[info_indices][:k_info] != info_bits)

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
