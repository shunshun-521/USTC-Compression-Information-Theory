"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归 PSCD 版本（高效实现）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """f 运算（box-plus，大 LLR 时退化为 min-sum）"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    result = np.empty(np.broadcast(La, Lb).shape, dtype=np.float64)
    it = np.nditer([La, Lb, result], flags=["refs_ok"], op_flags=[["readonly"], ["readonly"], ["writeonly"]])
    for a, b, out in it:
        av = float(a)
        bv = float(b)
        if abs(av) > 30 or abs(bv) > 30:
            out[...] = np.sign(av) * np.sign(bv) * min(abs(av), abs(bv))
        else:
            ta = np.tanh(av / 2.0)
            tb = np.tanh(bv / 2.0)
            out[...] = 2.0 * np.arctanh(np.clip(ta * tb, -0.999999, 0.999999))
    return result


def g_operation(La, Lb, u_hat):
    """g 运算：u=0 -> La+Lb, u=1 -> La-Lb"""
    u_hat = np.asarray(u_hat)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def permute_llr_for_decoder(llr_ch):
    """保留接口：PSCD 直接使用自然顺序 LLR"""
    return np.asarray(llr_ch, dtype=np.float64)


def _update_llrs(L, B, x, N, n):
    """Algorithm 4: 更新 LLR 树"""
    for j in range(n - 1, -1, -1):
        s = 1 << (n - j)
        t = s // 2
        for i in range(x, N, s):
            if t > (i % s):
                L[i, j] = f_operation(L[i, j + 1], L[i + t, j + 1])
            else:
                L[i, j] = g_operation(L[i, j + 1], L[i - t, j + 1], B[i - t, j])


def _update_bits(B, x, n):
    """Algorithm 5: 更新部分和数组"""
    b = [x]
    for j in range(n):
        s = 1 << (n - j)
        t = s // 2
        bnext = []
        for i in b:
            if t <= (i % s):
                B[i - t, j + 1] = int(B[i, j]) ^ int(B[i - t, j])
                B[i, j + 1] = B[i, j]
                bnext.extend([i, i - t])
        b = bnext


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 PSCD 译码。
    llr_ch: 自然信道顺序 LLR；frozen_bits: 1=冻结, 0=信息
    """
    llr = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)

    frozen_set = set(np.where(frozen_bits)[0])
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, n] = llr

    for phase in range(N):
        l = br[phase]
        _update_llrs(L, B, l, N, n)
        if l in frozen_set:
            B[l, 0] = 0
        else:
            B[l, 0] = 0 if L[l, 0] >= 0 else 1
        _update_bits(B, l, n)

    return B[:, 0].astype(int)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_block(block, offset):
        n = len(block)
        if n == 1:
            u_hat[offset] = 0 if frozen_bits[offset] or block[0] >= 0 else 1
            return
        half = n // 2
        left = f_operation(block[:half], block[half:])
        decode_block(left, offset)
        right = g_operation(block[:half], block[half:], u_hat[offset : offset + half])
        decode_block(right, offset + half)

    decode_block(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    br = bit_reversal_permutation(N)
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phase in range(N):
        l = br[phase]
        layers = list(range(n))
        llr_layer_vec.append(layers)
        bit_layers = list(range(n))
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec
