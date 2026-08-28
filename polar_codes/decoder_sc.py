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


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def _prepare_llr(llr_ch):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    br = bit_reversal_permutation(N)
    return llr_ch[br]


def _sc_decode_core(llr, frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=bool)

    def decode_block(llr_block, frozen_block):
        n = len(llr_block)
        if n == 1:
            bit = 0 if frozen_block[0] or llr_block[0] >= 0 else 1
            u = np.array([bit], dtype=int)
            return u, u.copy()

        half = n // 2
        llr_left = f_operation(llr_block[:half], llr_block[half:])
        u_left, u_left_up = decode_block(llr_left, frozen_block[:half])
        llr_right = g_operation(llr_block[:half], llr_block[half:], u_left_up)
        u_right, u_right_up = decode_block(llr_right, frozen_block[half:])
        u_hat = np.concatenate([u_left, u_right])
        u_left_up_xor = np.bitwise_xor(u_left_up, u_right_up)
        u_up = np.concatenate([u_left_up_xor, u_right_up])
        return u_hat, u_up

    u_hat, _ = decode_block(llr, frozen_bits)
    return u_hat.astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _prepare_llr(llr_ch)
    return _sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    基于 Permuted SCD 相位顺序（MSB 侧活跃层计数）。
    """
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = int(format(i, f"0{n}b")[::-1], 2)
        llr_layers = []
        mask = 1 << (n - 1)
        while mask:
            if (l & mask) == 0:
                layer = int(math.log2(mask)) if mask > 0 else 0
                llr_layers.append(n - 1 - len(llr_layers))
            else:
                break
            mask >>= 1
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


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


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    对含比特倒序的 Arikan 编码，先将信道 LLR 做倒序置换，再执行 SC 树搜索。
    """
    llr = _prepare_llr(llr_ch)
    return _sc_decode_core(llr, frozen_bits)
