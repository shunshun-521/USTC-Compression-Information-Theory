"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from encoder import polar_encode, build_generator_matrix
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder, crc_encode
from construction import ga_construction


def run_unit_tests():
    """各模块数值正确性校验"""
    # 编码器校验：蝶形结果与生成矩阵一致
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    G = build_generator_matrix(4)
    x_ref = (u @ G) % 2
    assert np.array_equal(x, x_ref), f"编码器错误: {x} vs {x_ref}"

    # SC 译码校验：高信噪比下应无错误
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    sigma = eb_n0_to_sigma(10.0, K / N)
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_hat = sc_decode(llr, frozen_bits)
        assert np.array_equal(u[info_idx], u_hat[info_idx]), "SC 高信噪比译码失败"

    # 路径度量校验：L=1 的 SCL 应等价于 SC
    llr_test = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
    u_sc = sc_decode(llr_test, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr_test)
    assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    print("单元测试全部通过。")


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
    """蒙特卡洛仿真"""
    rng = np.random.default_rng(seed)
    rate = K / N
    k_info = K - crc_length
    results = []

    if info_indices is None:
        info_idx, _, _ = ga_construction(N, K, 2.5)
    else:
        info_idx = np.asarray(info_indices)

    for eb_n0_db in eb_n0_db_list:
        sigma = eb_n0_to_sigma(eb_n0_db, rate)
        num_errors = 0
        num_bit_errors = 0
        num_frames = 0
        total_decode_time = 0.0
        total_iters = 0

        while num_frames < max_frames and num_errors < min_errors:
            u = np.zeros(N, dtype=int)
            if crc_length > 0:
                info_bits = rng.integers(0, 2, k_info)
                payload = crc_encode(info_bits, crc_length)
                u[info_idx[: len(payload)]] = payload
            else:
                u[info_idx] = rng.integers(0, 2, K)

            x = polar_encode(u)
            y = awgn_channel(bpsk_modulate(x), sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if decoder_type == "bp" and aux is not None:
                total_iters += aux

            if crc_length > 0:
                payload_hat = u_hat[info_idx[:K]]
                frame_err = not np.array_equal(payload_hat, u[info_idx[:K]])
                bit_err = np.sum(payload_hat[:k_info] != u[info_idx[:k_info]])
            else:
                frame_err = not np.array_equal(u_hat[info_idx], u[info_idx])
                bit_err = np.sum(u_hat[info_idx] != u[info_idx])

            num_frames += 1
            num_errors += int(frame_err)
            num_bit_errors += int(bit_err)

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
