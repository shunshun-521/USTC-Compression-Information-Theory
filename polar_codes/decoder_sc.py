"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np
from encoder import polar_encode


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, v_hat):
    """
    g 运算：g(La, Lb, v_hat) = (1 - 2*v_hat) * La + Lb
    v_hat 为左子树已译码比特的局部极化编码结果（非原始信息比特）。
    """
    return (1 - 2 * v_hat) * La + Lb


def _partial_encode(u_block):
    """对子块执行极化编码（与全局编码规则一致）。"""
    return polar_encode(u_block)


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码。
    参数：
        llr: 长度 N 的信道 LLR 数组
        frozen_bits: 长度 N 的 bool 数组，True 表示冻结位（置 0）
    返回：
        u_hat: 长度 N 的估计源序列
    """
    N = len(llr)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    u_hat = np.zeros(N, dtype=int)

    def decode_rec(llr_vec, stage, bit_idx):
        if stage == 0:
            if frozen_bits[bit_idx]:
                u_hat[bit_idx] = 0
            else:
                u_hat[bit_idx] = 0 if llr_vec[0] >= 0 else 1
            return

        half = 1 << (stage - 1)
        llr_left = f_operation(llr_vec[:half], llr_vec[half:])
        decode_rec(llr_left, stage - 1, bit_idx)

        u_left = u_hat[bit_idx:bit_idx + half]
        v_left = _partial_encode(u_left)
        llr_right = g_operation(llr_vec[:half], llr_vec[half:], v_left)
        decode_rec(llr_right, stage - 1, bit_idx + half)

    decode_rec(llr, int(math.log2(N)), 0)
    return u_hat


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        p = phi
        for layer in range(n):
            if p % 2 == 0:
                layers.append(layer)
                p //= 2
            else:
                break
        llr_layer_vec.append(layers)

        layers = []
        p = phi
        for layer in range(n):
            if p % 2 == 1:
                layers.append(layer)
            p //= 2
        bit_layer_vec.append(layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（显式栈实现，等价于递归版本）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(math.log2(N))
    u_hat = np.zeros(N, dtype=int)

    # 栈元素: (stage, bit_idx, llr_vec, phase)
    # phase: 0=进入左子树, 1=左子树完成准备右子树
    stack = [(n, 0, llr_ch.copy(), 0)]

    while stack:
        stage, bit_idx, llr_vec, phase = stack.pop()

        if stage == 0:
            if frozen_bits[bit_idx]:
                u_hat[bit_idx] = 0
            else:
                u_hat[bit_idx] = 0 if llr_vec[0] >= 0 else 1
            continue

        half = 1 << (stage - 1)
        if phase == 0:
            llr_left = f_operation(llr_vec[:half], llr_vec[half:])
            stack.append((stage, bit_idx, llr_vec, 1))
            stack.append((stage - 1, bit_idx, llr_left, 0))
        else:
            u_left = u_hat[bit_idx:bit_idx + half]
            v_left = _partial_encode(u_left)
            llr_right = g_operation(llr_vec[:half], llr_vec[half:], v_left)
            stack.append((stage - 1, bit_idx + half, llr_right, 0))

    return u_hat


if __name__ == "__main__":
    from construction import ga_construction
    from channel import bpsk_modulate, compute_llr, eb_n0_to_sigma

    u = np.array([1, 0, 1, 1])
    x = polar_encode(u)
    print("encode test:", x)
    assert np.array_equal(x, [1, 1, 0, 1]), f"编码器错误: {x}"

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info_idx] = False

    sigma = eb_n0_to_sigma(10.0, 0.5)
    errors = 0
    for seed in range(100):
        rng = np.random.default_rng(seed)
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = compute_llr(bpsk_modulate(polar_encode(u)) + rng.normal(0, sigma, N), sigma)

        u1 = sc_decode(llr, frozen)
        u2 = sc_decode_recursive(llr, frozen)
        assert np.array_equal(u1, u2), "SC recursive/non-recursive mismatch"
        if not np.array_equal(u[info_idx], u1[info_idx]):
            errors += 1
    print(f"SC test N=64: {errors}/100 frame errors at Eb/N0=10dB")
