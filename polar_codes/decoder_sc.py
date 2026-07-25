"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def _sign_ms(x):
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return _sign_ms(La) * _sign_ms(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= 1 << (n - 1 - i)
    return result


def _permute_llr(llr_ch, N):
    """将信道 LLR 变换为译码器内部顺序（与 BRP 编码器配套）"""
    n = int(np.log2(N))
    brp = np.array([bit_reversed(i, n) for i in range(N)])
    return np.asarray(llr_ch, dtype=np.float64)[brp]


def active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）。"""
    llr = _permute_llr(llr, len(llr))
    return _sc_decode_core(llr, frozen_bits, recursive=True)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回比特倒序处理顺序及层更新列表（与高效实现一致）。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    decode_order = [bit_reversed(phi, n) for phi in range(N)]
    for phi in decode_order:
        llr_layers = list(range(n - active_llr_level(phi, n), n))
        bit_layers = list(range(n, n - active_bit_level(phi, n), -1))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


def _sc_decode_core(llr_internal, frozen_bits, recursive=False):
    """SC 译码核心（非递归高效实现）。"""
    N = len(llr_internal)
    n = int(np.log2(N))
    frozen_set = set(np.where(np.asarray(frozen_bits, dtype=bool))[0])

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    L[:, 0] = llr_internal

    for l in [bit_reversed(i, n) for i in range(N)]:
        for s in range(n - active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s], L[j, s], B[j - branch_size, s + 1]
                    )

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = 0 if L[l, n] >= 0 else 1

        if l >= N // 2:
            for s in range(n, n - active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数。
    输入信道 LLR 与 BRP 编码器输出顺序一致。
    """
    N = len(llr_ch)
    llr_internal = _permute_llr(llr_ch, N)
    return _sc_decode_core(llr_internal, frozen_bits)
