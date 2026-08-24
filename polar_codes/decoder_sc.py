"""
极化码 SC（串行抵消）译码器
PSCD 非递归实现，比特倒序相位译码
"""
import numpy as np

from encoder import bit_reversed


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


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


def _update_llrs(L, B, phase, n, N):
    start = n - _active_llr_level(phase, n)
    for s in range(start, n):
        block = 1 << (s + 1)
        half = block // 2
        for j in range(phase, N, block):
            if j % block < half:
                L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
            else:
                L[j, s + 1] = g_operation(
                    L[j - half, s], L[j, s], B[j - half, s + 1]
                )


def _update_bits(B, phase, n, N):
    if phase < N // 2:
        return
    stop = n - _active_bit_level(phase, n)
    for s in range(n, stop, -1):
        block = 1 << s
        half = block // 2
        for j in range(phase, -1, -block):
            if j % block >= half:
                B[j - half, s - 1] = B[j, s] ^ B[j - half, s]
                B[j, s - 1] = B[j, s]


def sc_decode(llr_ch, frozen_bits):
    """非递归 PSCD 译码"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch

    for phase in [bit_reversed(i, n) for i in range(N)]:
        _update_llrs(L, B, phase, n, N)
        B[phase, n] = 0 if frozen_bits[phase] else (0 if L[phase, n] >= 0 else 1)
        _update_bits(B, phase, n, N)

    return B[:, n].copy()


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（小规模参考）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    u_hat = np.zeros(N, dtype=np.int8)

    def decode_node(node, offset):
        m = len(node)
        if m == 1:
            idx = offset
            u_hat[idx] = 0 if frozen_bits[idx] or node[0] >= 0 else 1
            return
        half = m // 2
        decode_node(f_operation(node[:half], node[half:]), offset)
        decode_node(
            g_operation(node[:half], node[half:], u_hat[offset : offset + half]),
            offset + half,
        )

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """供 SCL 使用的层索引"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        phase = bit_reversed(i, n)
        start = n - _active_llr_level(phase, n)
        llr_layer_vec.append(list(range(start, n)))
        if phase < N // 2:
            bit_layer_vec.append([])
        else:
            stop = n - _active_bit_level(phase, n)
            bit_layer_vec.append(list(range(n, stop, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec
