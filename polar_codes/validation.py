"""模块数值正确性校验（各实验脚本导入）。"""
import numpy as np

from construction import ga_construction
from encoder import polar_encode
from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma
from decoder_sc import sc_decode
from decoder_scl import SCLDecoder


def run_unit_tests():
    # 编码器：与生成矩阵 G=B_N F^{⊗n} 手算一致（N=4, u=[1,0,1,1] -> x=[1,1,0,1]）
    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False
    sigma = eb_n0_to_sigma(10.0, K / N)
    err = 0
    for _ in range(100):
        payload = np.random.randint(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        if not np.array_equal(sc_decode(llr, frozen)[info], payload):
            err += 1
    assert err == 0, f"SC 无损译码失败: {err}/100 帧错误"

    # L=1 的 SCL 应等价于 SC
    for _ in range(20):
        payload = np.random.randint(0, 2, size=K)
        u = np.zeros(N, dtype=int)
        u[info] = payload
        llr = compute_llr(bpsk_modulate(polar_encode(u)), sigma)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL(L=1) 与 SC 不一致"
