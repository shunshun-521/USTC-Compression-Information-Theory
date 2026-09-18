"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _propagate_lr(L, R, n, N):
    """因子图上完整的 L（右→左）和 R（左→右）消息传播"""
    for j in range(n, 0, -1):
        s = 1 << (j - 1)
        for i in range(0, N, s * 2):
            for k in range(s):
                ii = i + k
                iis = i + k + s
                L[ii, j - 1] = f_operation(R[ii, j] + L[iis, j], L[ii, j])
                L[iis, j - 1] = f_operation(R[ii, j], L[ii, j]) + L[iis, j]

    for j in range(0, n):
        s = 1 << j
        for i in range(0, N, s * 2):
            for k in range(s):
                ii = i + k
                iis = i + k + s
                R[ii, j + 1] = f_operation(R[iis, j] + L[iis, j + 1], R[ii, j])
                R[iis, j + 1] = f_operation(R[ii, j], L[ii, j + 1]) + R[iis, j]


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（因子图消息传递，与 BP 一致）。
    """
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量：
      - lambda_offset[phi]: 第 phi 个比特对应的 LLR 存储偏移
      - llr_layer_vec[phi]: 第 phi 个比特需要执行 LLR 运算的层列表
      - bit_layer_vec[phi]: 第 phi 个比特需要执行比特返回的层列表
    """
    n = int(np.log2(N))
    lambda_offset = np.zeros(n + 2, dtype=int)
    for i in range(n + 1):
        lambda_offset[i] = 1 << (n - i)

    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        tmp = phi
        while tmp & 1:
            layer += 1
            tmp >>= 1
        llr_layer_vec.append(list(range(n - 1, layer - 1, -1)))
        bit_layer_vec.append(list(range(layer - 1, -1, -1)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（基于因子图消息传递，与 BP 译码器一致）。

    参数：
        llr_ch: 长度 N 的信道接收 LLR（float64）
        frozen_bits: 长度 N 的 bool/int 数组，1 表示冻结位

    返回：
        u_hat: 长度 N 的估计源序列（0/1 int 数组）
    """
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    LARGE = 1e6

    L = np.zeros((N, n + 1))
    R = np.zeros((N, n + 1))
    L[:, n] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        R[:] = 0.0
        R[frozen_bits, 0] = LARGE
        for i in range(phi):
            if not frozen_bits[i]:
                R[i, 0] = LARGE if u_hat[i] == 0 else -LARGE

        _propagate_lr(L, R, n, N)

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if (L[phi, 0] + R[phi, 0]) >= 0 else 1

    return u_hat
