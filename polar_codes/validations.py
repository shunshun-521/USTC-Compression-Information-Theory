"""数值正确性校验"""
import numpy as np
from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_validations():
    """运行全部单元测试，失败时抛出 AssertionError"""
    # 编码器校验
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    F = np.array([[1, 1], [0, 1]], dtype=int)
    G = F
    for _ in range(int(np.log2(len(u))) - 1):
        G = np.kron(G, F)
    assert np.array_equal(x, (u @ G) % 2), f"编码器错误: {x}"

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0
    frozen_bool = frozen_bits.astype(bool)

    rng = np.random.default_rng(123)
    sigma = eb_n0_to_sigma(10.0, K / N)
    for _ in range(100):
        u_sent = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_hat = sc_decode(llr, frozen_bool)
        assert np.array_equal(u_hat[info_idx], payload), "SC 无损译码失败"

    # 单路径 SCL 应等价于 SC
    scl = SCLDecoder(N, frozen_bool, list_size=1, crc_length=0)
    for _ in range(20):
        u_sent = np.zeros(N, dtype=int)
        payload = rng.integers(0, 2, K)
        u_sent[info_idx] = payload
        x = polar_encode(u_sent)
        llr = compute_llr(bpsk_modulate(x), sigma)
        u_sc = sc_decode(llr, frozen_bool)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不一致"

    print("全部单元测试通过。")


if __name__ == "__main__":
    run_validations()
