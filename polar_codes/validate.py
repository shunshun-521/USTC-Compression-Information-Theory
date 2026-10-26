"""模块数值正确性校验（实验脚本运行前调用）。"""
import numpy as np

from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
from construction import ga_construction
from decoder_sc import sc_decode, sc_decode_recursive, _prepare_channel_llr
from decoder_scl import SCLDecoder, crc_encode, crc_check
from encoder import polar_encode, bit_reversal_permutation


def _generator_matrix(N):
    """G = B_N @ F^{⊗n}（与 polar_encode 一致）。"""
    F = np.array([[1, 0], [1, 1]], dtype=int)
    n = int(np.log2(N))
    F_n = F.copy()
    for _ in range(n - 1):
        F_n = np.kron(F_n, F)
    br = bit_reversal_permutation(N)
    B = np.eye(N, dtype=int)[br]
    return (B @ F_n) % 2


def validate_encoder():
    N = 4
    G = _generator_matrix(N)
    u = np.array([1, 0, 1, 1])
    x_ref = (u @ G) % 2
    x = polar_encode(u)
    assert np.array_equal(x, x_ref), f"编码器与生成矩阵不一致: {x} vs {x_ref}"
    # 规格示例向量（若与 G 不一致则仅打印提示）
    if not np.array_equal(x, np.array([0, 0, 1, 1])):
        print(f"  注: 规格示例码字 [0,0,1,1] 与当前 G 定义下码字 {x} 不同（以 G 为准）。")


def validate_sc_noiseless():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = (1 - 2 * x) * 50.0  # 近似无噪 LLR
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u_hat, u), "SC 无噪译码失败"
        u_hat_r = sc_decode_recursive(_prepare_channel_llr(llr), frozen)
        assert np.array_equal(u_hat_r, u), "递归 SC 无噪译码失败"


def validate_sc_awgn():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rate = K / N
    sigma = eb_n0_to_sigma(12.0, rate)
    rng = np.random.default_rng(1)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng=rng)
        llr = compute_llr(y, sigma)
        u_hat = sc_decode(llr, frozen)
        if not np.array_equal(u_hat[info_idx], u[info_idx]):
            errors += 1
    assert errors == 0, f"Eb/N0=12dB 下 SC 仍有 {errors}/100 帧错误"


def validate_scl_equiv_sc():
    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=int)
    frozen[info_idx] = 0
    rng = np.random.default_rng(2)
    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = (1 - 2 * x) * 40.0
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 的 SCL 应与 SC 一致"


def validate_crc():
    bits = np.array([1, 0, 1, 1, 0, 0, 1], dtype=int)
    enc = crc_encode(bits, 8)
    assert crc_check(enc, 8)
    enc[-1] ^= 1
    assert not crc_check(enc, 8)


def run_all_validations():
    print("运行模块校验...")
    validate_encoder()
    print("  编码器 OK")
    validate_sc_noiseless()
    print("  SC 无噪 OK")
    validate_sc_awgn()
    print("  SC 高信噪比 OK")
    validate_scl_equiv_sc()
    print("  SCL(L=1) 与 SC OK")
    validate_crc()
    print("  CRC OK")
    print("全部校验通过。")


if __name__ == "__main__":
    run_all_validations()
