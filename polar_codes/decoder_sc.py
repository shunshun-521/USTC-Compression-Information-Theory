"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """精确 log 域 f 运算"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.logaddexp(0.0, La + Lb) - np.logaddexp(La, Lb)


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat, dtype=np.float64)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return Lb + (1.0 - 2.0 * u_hat) * La


def f_min_sum(La, Lb):
    """min-sum 近似 f（BP 译码使用）"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（树形遍历）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, bit_offset, length):
        if length == 1:
            idx = bit_offset
            if frozen_bits[idx]:
                u_hat[idx] = 0
            else:
                u_hat[idx] = 0 if llr_node[0] >= 0 else 1
            return np.array([u_hat[idx]], dtype=np.int8)

        half = length // 2
        upper = f_operation(llr_node[:half], llr_node[half:])
        beta_u = decode_node(upper, bit_offset, half)
        lower = g_operation(llr_node[:half], llr_node[half:], beta_u)
        beta_l = decode_node(lower, bit_offset + half, half)
        return np.concatenate([(beta_u + beta_l) % 2, beta_l])

    decode_node(llr, 0, N)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = 1 << (i - 1)

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        pm2 = phi
        layer = 0
        while pm2 % 2 == 1:
            layers_llr.append(layer)
            pm2 >>= 1
            layer += 1
        layers_llr.append(layer)
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        if phi % 2 == 0:
            layers_bit = list(range(n))
        else:
            pm2 = phi
            layer = 0
            while pm2 % 2 == 1:
                layers_bit.append(layer)
                pm2 >>= 1
                layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def _f_layers(P, layer, lambda_offset):
    step = lambda_offset[layer + 1]
    N = len(P[layer])
    for block in range(0, N, 2 * step):
        La = P[layer + 1][block : block + step]
        Lb = P[layer + 1][block + step : block + 2 * step]
        P[layer][block : block + step] = f_operation(La, Lb)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码（分层存储）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = [np.zeros(N, dtype=np.float64) for _ in range(n + 1)]
    C = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]
    P[n][:] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        if phi == 0:
            for layer in range(n - 1, -1, -1):
                _f_layers(P, layer, lambda_offset)
        else:
            top = llr_layer_vec[phi][-1]
            for layer in range(n - 1, top - 1, -1):
                _f_layers(P, layer, lambda_offset)

            pm2 = phi
            layer_idx = 0
            while pm2 % 2 == 1:
                block = (pm2 >> 1) * lambda_offset[layer_idx + 1]
                step = lambda_offset[layer_idx + 1]
                u_left = C[layer_idx + 1][block : block + step]
                La = P[layer_idx + 1][block : block + step]
                Lb = P[layer_idx + 1][block + step : block + 2 * step]
                P[layer_idx][block + step : block + 2 * step] = g_operation(La, Lb, u_left)
                pm2 >>= 1
                layer_idx += 1

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0][phi] >= 0 else 1

        C[0][phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            step = lambda_offset[layer + 1]
            for block in range(0, N, 2 * step):
                left = C[layer + 1][block : block + step]
                right = C[layer + 1][block + step : block + 2 * step]
                C[layer][block : block + step] = (left + right) % 2
                C[layer][block + step : block + 2 * step] = right

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """SC 译码入口（默认使用已验证的递归实现）"""
    return sc_decode_recursive(llr_ch, frozen_bits)
