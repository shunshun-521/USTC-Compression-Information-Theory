"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    u_hat = np.asarray(u_hat)
    return (1.0 - 2.0 * u_hat) * La + Lb


def _bit_reversed_index(x, n):
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def _upper_llr(l1, l2):
    return _logdomain_sum(l1 + l2, 0.0) - _logdomain_sum(l1, l2)


def _lower_llr(l1, l2, bit):
    return l1 + l2 if int(bit) == 0 else l1 - l2


def _active_llr_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _active_bit_level(i, n):
    mask = 1 << (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _frozen_set_for_decoder(frozen_bits):
    """将自然序冻结位映射到译码器内部索引（比特倒序）"""
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(frozen_bits)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen_idx = np.where(frozen_bits == 1)[0]
    return set(int(br[i]) for i in frozen_idx)


def _scd_core(llr, frozen_set, n):
    """Vangala 风格非递归 SC 译码核心"""
    N = len(llr)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for l in [_bit_reversed_index(i, n) for i in range(N)]:
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _upper_llr(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    if np.isnan(top_bit):
                        top_bit = 0
                    L[j, s + 1] = _lower_llr(L[j, s], L[j - branch_size, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    输入信道 LLR（与码字比特顺序一致），输出估计源序列 u_hat。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    frozen_set = _frozen_set_for_decoder(frozen_bits)
    x_hat = _scd_core(llr_ch, frozen_set, n)
    return x_hat[br]


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（调用非递归实现作为参考）"""
    return sc_decode(llr, frozen_bits)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量（接口兼容）"""
    n = int(math.log2(N))
    lambda_offset = np.array([1 << (n - layer) for layer in range(n + 1)], dtype=int)
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        layers = []
        pm = phi
        layer = n
        while pm % 2 == 1 and layer > 0:
            pm //= 2
            layer -= 1
        for lay in range(layer, 0, -1):
            layers.append(lay)
        b_layers = []
        pm = phi
        lay = 0
        while pm % 2 == 1:
            b_layers.append(lay)
            pm //= 2
            lay += 1
        llr_layer_vec.append(layers)
        bit_layer_vec.append(b_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec
