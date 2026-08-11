"""单元测试与验证函数"""
import numpy as np

from encoder import polar_encode
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder
from construction import ga_construction


def run_unit_tests():
    """运行所有模块的数值正确性校验。"""
    # 编码器校验：u=[1,0,1,1] -> x=[1,1,0,1]（G_N 手算验证）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    expected = np.array([1, 1, 0, 1])
    assert np.array_equal(x, expected), f"编码器错误: {x}, 期望 {expected}"

    # SC 译码校验（极低噪声）
    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    errors = 0
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u)
        llr = np.where(x == 0, 100.0, -100.0)
        u_hat = sc_decode(llr, frozen_bits)
        if not np.array_equal(u[info_idx], u_hat[info_idx]):
            errors += 1
    assert errors == 0, f"SC 无损译码失败: {errors}/100 帧错误"

    # 路径度量校验：L=1 SCL 等价于 SC
    N4, K4 = 4, 2
    info4, _, _ = ga_construction(N4, K4, 2.5)
    frozen4 = np.ones(N4, dtype=int)
    frozen4[info4] = 0
    u4 = np.zeros(N4, dtype=int)
    u4[info4] = [1, 0]
    llr4 = np.where(polar_encode(u4) == 0, 50.0, -50.0)
    u_sc = sc_decode(llr4, frozen4)
    u_scl, _ = SCLDecoder(N4, frozen4, list_size=1).decode(llr4)
    assert np.array_equal(u_sc, u_scl), "L=1 SCL 与 SC 不等价"

    print("所有单元测试通过。")


if __name__ == "__main__":
    run_unit_tests()
