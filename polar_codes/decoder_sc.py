"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（逐比特 LLR 递推，与编码端一致）
"""
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算（和接近 0 时用 Lb 稳定）"""
    g = (1 - 2 * u_hat) * La + Lb
    if np.ndim(g) == 0:
        if abs(g) < 1e-12:
            return float(Lb)
        return float(g)
    mask = np.abs(g) < 1e-12
    return np.where(mask, Lb, g)


def _llr_at_bit(channel_llr_br, u_prefix, phi):
    """计算第 phi 个比特的 LLR（channel_llr_br 已为比特倒序后的信道 LLR）"""
    N = len(channel_llr_br)

    def rec(llr, base, n, target):
        if n == 1:
            return float(llr[0])
        h = n // 2
        if target < base + h:
            ll = f_operation(llr[:h], llr[h:])
            return rec(ll, base, h, target)
        u_l = u_prefix[base : base + h]
        lr = g_operation(llr[:h], llr[h:], u_l)
        return rec(lr, base + h, h, target)

    return rec(channel_llr_br, 0, N, phi)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_sub(llr_sub, frozen_sub, offset):
        n = len(llr_sub)
        if n == 1:
            if frozen_sub[0]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = 0 if llr_sub[0] >= 0 else 1
            return
        half = n // 2
        llr_left = f_operation(llr_sub[:half], llr_sub[half:])
        decode_sub(llr_left, frozen_sub[:half], offset)
        u_left = u_hat[offset : offset + half]
        llr_right = g_operation(llr_sub[:half], llr_sub[half:], u_left)
        decode_sub(llr_right, frozen_sub[half:], offset + half)

    decode_sub(llr, frozen_bits, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供 SCL 等模块使用）"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        for layer in range(n):
            if ((phi >> layer) & 1) == 0:
                layers_llr.append(layer)
            else:
                break
        llr_layer_vec.append(layers_llr)
        if (phi & 1) == 0:
            bit_layer_vec.append(list(range(n)))
        else:
            layers_bit = []
            layer = 0
            while layer < n and ((phi >> layer) & 1) == 1:
                layers_bit.append(layer)
                layer += 1
            bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码主接口（BPSK-AWGN，LLR = 2y/σ²）。

    对当前编码约定（蝶形 + 比特倒序，且编码为自逆变换），
    硬判决后经 polar_encode 等价于无噪 SC 的逆变换；再叠加软度量
    的逐比特 SC 用于短码长。长码长下以硬判决逆变换为主并强制冻结位为 0。
    """
    from encoder import polar_encode

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)

    rev = bit_reversal_permutation(N)
    y = llr_ch[rev]
    u_hat = np.zeros(N, dtype=int)
    for phi in range(N):
        llr_phi = _llr_at_bit(y, u_hat, phi)
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if llr_phi >= 0 else 1
    return u_hat
