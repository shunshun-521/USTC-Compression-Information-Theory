"""
蒙特卡洛仿真主循环
"""
import time
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from utils import crc_encode


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
    info_indices=None,
    frozen_bits=None,
):
    """
    蒙特卡洛仿真。
    """
    rng = np.random.default_rng(seed)
    rate = K / N
    K_info = K - crc_length

    if frozen_bits is None:
        if info_indices is None:
            info_indices, _, _ = ga_construction(N, K, 2.5)
        frozen_bits = np.ones(N, dtype=int)
        frozen_bits[info_indices] = 0
    else:
        info_indices = np.where(frozen_bits == 0)[0]

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
                info_payload = rng.integers(0, 2, K_info)
                info_with_crc = crc_encode(info_payload, crc_length)
                u = np.zeros(N, dtype=int)
                u[info_indices[:K]] = info_with_crc
            else:
                u = np.zeros(N, dtype=int)
                u[info_indices] = rng.integers(0, 2, K)

            x = polar_encode(u)
            s = bpsk_modulate(x)
            y = awgn_channel(s, sigma, rng)
            llr = compute_llr(y, sigma)

            t0 = time.perf_counter()
            u_hat, aux = decoder(llr)
            total_decode_time += time.perf_counter() - t0

            if aux is not None and decoder_type == "bp":
                total_iters += aux

            if crc_length > 0:
                err = not np.array_equal(u_hat[info_indices[:K]], u[info_indices[:K]])
                bit_err = np.sum(u_hat[info_indices[:K_info]] != u[info_indices[:K_info]])
            else:
                err = not np.array_equal(u_hat[info_indices], u[info_indices])
                bit_err = np.sum(u_hat[info_indices] != u[info_indices])

            num_errors += int(err)
            num_bit_errors += int(bit_err)
            num_frames += 1

        bler = num_errors / num_frames
        ber = num_bit_errors / (num_frames * K_info) if K_info > 0 else 0.0
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


def run_unit_tests():
    """模块正确性校验"""
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 0, 1, 1]), f"编码器错误: {x}"

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        u_hat = sc_decode(llr, frozen)
        errors += int(not np.array_equal(u_hat[info_idx], u[info_idx]))
    assert errors == 0, f"SC 无损校验失败: {errors}/100 帧错误"

    from decoder_scl import SCLDecoder
    scl1 = SCLDecoder(N, frozen, list_size=1)
    errors_scl = 0
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), eb_n0_to_sigma(10.0, K / N))
        uh, _ = scl1.decode(llr)
        errors_scl += int(not np.array_equal(uh[info_idx], u[info_idx]))
    assert errors_scl == 0, f"SCL L=1 校验失败: {errors_scl}/20"

    print("单元测试通过。")
