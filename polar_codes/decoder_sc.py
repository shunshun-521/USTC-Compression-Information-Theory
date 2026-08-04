"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
基于置换 SC 译码结构（Vangala et al.）
"""
import math
import numpy as np

LLR_CLIP = 100.0

# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(x, n):
    """比特倒序索引"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def _active_llr_level(i, n):
    """LLR 更新起始层"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    """比特回传起始层"""
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _logdomain_sum(x, y):
    """对数域加法"""
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    """f 运算（min-sum，带限幅）"""
    l1 = np.clip(l1, -LLR_CLIP, LLR_CLIP)
    l2 = np.clip(l2, -LLR_CLIP, LLR_CLIP)
    return np.clip(f_operation(l1, l2), -LLR_CLIP, LLR_CLIP)


def _lower_llr(l1, l2, b):
    """g 运算（带限幅）"""
    l1 = np.clip(l1, -LLR_CLIP, LLR_CLIP)
    l2 = np.clip(l2, -LLR_CLIP, LLR_CLIP)
    if int(b) == 0:
        return np.clip(l1 + l2, -LLR_CLIP, LLR_CLIP)
    return np.clip(l1 - l2, -LLR_CLIP, LLR_CLIP)


def _update_llrs(L, B, l, n, N):
    """更新 LLR 树"""
    for s in range(n - _active_llr_level(l, n), n):
        block_size = 2 ** (s + 1)
        branch_size = block_size // 2
        for j in range(l, N, block_size):
            if j % block_size < branch_size:
                top_llr = L[j, s]
                btm_llr = L[j + branch_size, s]
                L[j, s + 1] = _upper_llr(top_llr, btm_llr)
            else:
                btm_llr = L[j, s]
                top_llr = L[j - branch_size, s]
                top_bit = B[j - branch_size, s + 1]
                L[j, s + 1] = _lower_llr(btm_llr, top_llr, top_bit)


def _update_bits(B, l, n, N):
    """比特回传"""
    if l < N / 2:
        return
    for s in range(n, n - _active_bit_level(l, n), -1):
        block_size = 2 ** s
        branch_size = block_size // 2
        for j in range(l, -1, -block_size):
            if j % block_size >= branch_size:
                B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                B[j, s - 1] = B[j, s]


def _sc_decode_core(llr_ch, frozen_bits):
    """置换 SC 译码核心"""
    N = len(llr_ch)
    n = int(math.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    llr_ch = np.clip(np.asarray(llr_ch, dtype=np.float64), -LLR_CLIP, LLR_CLIP)

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.float64)
    L[:, 0] = llr_ch

    for l in [_bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, l, n, N)
        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1
        _update_bits(B, l, n, N)

    return B[:, n].astype(np.int8)


# ==================== 对外接口 ====================


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return _sc_decode_core(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（与 sc_decode 等价）"""
    return _sc_decode_core(llr, frozen_bits)


def sc_decode_sequential(llr_ch, frozen_bits):
    """顺序 SC 译码（与 sc_decode 等价）"""
    return _sc_decode_core(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 辅助向量（供扩展实现使用）"""
    n = int(math.log2(N))
    lambda_offset = [0] * (n + 1)
    for layer in range(1, n + 1):
        lambda_offset[layer] = lambda_offset[layer - 1] + (1 << (n - layer + 1))

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers_llr = []
        psi = phi
        layer = 0
        while layer < n:
            if psi % 2 == 0:
                layers_llr.append(layer)
            psi //= 2
            layer += 1
        llr_layer_vec.append(layers_llr)

        layers_bit = []
        psi = phi
        layer = 0
        while layer < n:
            if psi % 2 == 1:
                layers_bit.append(layer)
            psi //= 2
            layer += 1
        bit_layer_vec.append(layers_bit)

    return lambda_offset, llr_layer_vec, bit_layer_vec
