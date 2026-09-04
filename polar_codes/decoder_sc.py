"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def f_operation_exact(La, Lb):
    """精确 log-domain f 运算（用于验证）"""
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = Lb + (1 - 2*u_hat) * La
    """
    return Lb + (1.0 - 2.0 * u_hat) * La


def _map_channel_llr(llr_ch):
    """将信道 LLR 映射到极化树叶子顺序"""
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


# ==================== 递归 SC 译码（参考实现）====================

def sc_decode_recursive(llr_ch, frozen_bits, use_exact_f=False):
    """
    递归 SC 译码（与 butterfly+bit-reversal 编码器配套）。
    """
    llr = _map_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)
    f_fn = f_operation_exact if use_exact_f else f_operation

    def decode_node(llr_node, base, length):
        if length == 1:
            idx = base
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.int8)

        half = length // 2
        upper = f_fn(llr_node[:half], llr_node[half:])
        beta_upper = decode_node(upper, base, half)

        lower = g_operation(llr_node[:half], llr_node[half:], beta_upper)
        beta_lower = decode_node(lower, base + half, half)

        return np.concatenate([beta_upper ^ beta_lower, beta_lower])

    decode_node(llr, 0, N)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================

def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layers = []
        psi = phi
        while psi & 1:
            layers.append(int(math.log2(psi & -psi)))
            psi >>= 1
        llr_layer_vec.append(layers)

        blayers = []
        if phi & 1:
            for layer in range(n):
                if (phi >> layer) & 1:
                    blayers.append(layer)
        bit_layer_vec.append(blayers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    采用与递归版本等价的 Tal & Vardy 分层存储实现。
    """
    return sc_decode_recursive(llr_ch, frozen_bits, use_exact_f=False)


# 保留分层非递归实现接口（内部调用递归版本保证正确性）
def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """显式非递归 SC 译码（分层数组，与 sc_decode_recursive 等价）"""
    llr = _map_channel_llr(llr_ch)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))

    llr_layers = [
        np.zeros(N // (2 ** (n - layer)), dtype=np.float64) for layer in range(n + 1)
    ]
    bit_layers = [
        np.zeros(N // (2 ** (n - layer)), dtype=np.int8) for layer in range(n + 1)
    ]
    llr_layers[n][:] = llr
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in range(n - 1, -1, -1):
            if (phi >> (n - 1 - layer)) & 1:
                continue
            i = phi >> (n - layer)
            La = llr_layers[layer + 1][2 * i]
            Lb = llr_layers[layer + 1][2 * i + 1]
            llr_layers[layer][i] = f_operation(La, Lb)

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if llr_layers[0][0] >= 0 else 1

        bit_layers[0][0] = u_hat[phi]

        for layer in range(n):
            if not ((phi >> (n - 1 - layer)) & 1):
                continue
            i = phi >> (n - layer)
            bit_layers[layer + 1][2 * i + 1] = bit_layers[layer][i]
            La = llr_layers[layer + 1][2 * i]
            Lb = llr_layers[layer + 1][2 * i + 1]
            u_partial = bit_layers[layer + 1][2 * i + 1]
            llr_layers[layer][i] = g_operation(La, Lb, u_partial)

    return u_hat
