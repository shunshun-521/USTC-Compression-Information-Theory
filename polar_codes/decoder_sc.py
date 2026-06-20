"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
  """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    result = 0
    for b in range(n):
        if (i >> b) & 1:
            result |= 1 << (n - 1 - b)
    return result


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


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_rec(llr_in, idx_start, length):
        if length == 1:
            if frozen_bits[idx_start]:
                u_hat[idx_start] = 0
            else:
                u_hat[idx_start] = 0 if llr_in[0] >= 0 else 1
            return

        half = length // 2
        llr_left = f_operation(llr_in[:half], llr_in[half:])
        decode_rec(llr_left, idx_start, half)

        u_left = u_hat[idx_start:idx_start + half]
        llr_right = g_operation(llr_in[:half], llr_in[half:], u_left)
        decode_rec(llr_right, idx_start + half, half)

    decode_rec(llr, 0, N)
    return u_hat


def precompute_sc_indices(N):
    """保留接口：返回比特倒序译码顺序及辅助参数。"""
    n = int(np.log2(N))
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    lambda_offset = [1 << i for i in range(n + 1)]
    return lambda_offset, decode_order, n


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（Permuted SC 结构）。
    llr_ch[i] 对应编码后码字第 i 个位置（含比特倒序置换）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))

    # 将信道 LLR 映射到译码树自然索引
    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int32)
    for j in range(N):
        L[j, 0] = llr_ch[_bit_reversed(j, n)]

    frozen_set = set(np.where(frozen_bits == 1)[0])

    for phi in range(N):
        l = _bit_reversed(phi, n)

        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size >> 1
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
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 1 << s
                branch_size = block_size >> 1
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = B[j, s] ^ B[j - branch_size, s]
                        B[j, s - 1] = B[j, s]

    u_hat = B[:, n].astype(int)
    return u_hat
