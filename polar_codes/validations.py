"""模块数值校验（各实验脚本启动时调用）"""
import numpy as np
from encoder import polar_encode
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from construction import ga_construction


def run_validations(verbose=True):
    """运行编码器 / SC / SCL 路径度量校验"""
    # 编码器：往返一致性
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    llr = (1 - 2 * x).astype(float) * 100.0
    u_hat = sc_decode(llr, np.zeros(len(u), dtype=bool))
    assert np.array_equal(u, u_hat), f"编码/SC 往返失败: u={u}, u_hat={u_hat}"

    # SC 高信噪比无损（N=64, K=32，无噪信道）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False
    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = (1 - 2 * x).astype(float) * 100.0
        u_hat = sc_decode(llr, frozen)
        assert np.array_equal(u[info_idx], u_hat[info_idx]), "高 SNR SC 译码失败"

    # SCL L=1 等价 SC
    u_test = np.zeros(N, dtype=int)
    u_test[info_idx] = rng.integers(0, 2, K)
    llr_t = (1 - 2 * polar_encode(u_test)).astype(float) * 100.0
    us = sc_decode(llr_t, frozen)
    usl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr_t)
    assert np.array_equal(us, usl), "SCL(L=1) 与 SC 不一致"

    if verbose:
        print("所有模块校验通过。")


if __name__ == "__main__":
    run_validations()
