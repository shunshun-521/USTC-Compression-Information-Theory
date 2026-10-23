"""
极化码 SC（串行抵消）译码器
置换 SC（Permuted SCD），与 mcba1n 风格编码配套
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * np.asarray(u_hat)) * La + Lb


def _bit_reversed(x, n):
    return int(f"{x:0{n}b}"[::-1], 2)


def _active_llr_level(i, n):
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
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
            mask >>= 1
        else:
            break
    return min(count, n)


def _hard_decision(llr):
    return 0 if llr >= 0 else 1


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC（自然顺序，用于交叉验证）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    u_hat = np.zeros(N, dtype=int)

    def decode_node(llr_node, offset):
        m = len(llr_node)
        if m == 1:
            if frozen_bits[offset]:
                u_hat[offset] = 0
            else:
                u_hat[offset] = _hard_decision(llr_node[0])
            return
        half = m // 2
        decode_node(f_operation(llr_node[:half], llr_node[half:]), offset)
        u_left = u_hat[offset:offset + half]
        decode_node(g_operation(llr_node[:half], llr_node[half:], u_left), offset + half)

    decode_node(llr, 0)
    return u_hat


def precompute_sc_indices(N):
    """预计算非递归 SC 的层索引（与置换 SCD 一致）"""
    n = int(np.log2(N))
    lambda_offset = [1 << l for l in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for i in range(N):
        l = _bit_reversed(i, n)
        start = n - _active_llr_level(l, n)
        llr_layer_vec.append(list(range(start, n)))
        bit_start = n - _active_bit_level(l, n) + 1
        if l < N // 2:
            bit_layer_vec.append([])
        else:
            bit_layer_vec.append(list(range(n, bit_start - 1, -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归置换 SC 译码（主实现）。
    L[i,0] = 信道 LLR；按 bit_reversed(phi, n) 顺序判决。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))

    L = np.zeros((N, n + 1), dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=np.int8)
    L[:, 0] = llr_ch
    frozen_set = set(np.where(frozen_bits)[0])
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        l = _bit_reversed(phi, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    top_bit = B[j - branch_size, s + 1]
                    L[j, s + 1] = g_operation(L[j - branch_size, s], L[j, s], top_bit)

        if l in frozen_set:
            B[l, n] = 0
        else:
            B[l, n] = _hard_decision(L[l, n])
        u_hat[l] = B[l, n]

        if l < N // 2:
            continue
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    return u_hat


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr

    u = np.array([1, 0, 1, 1])
    assert np.array_equal(polar_encode(u), [1, 1, 0, 1])

    for N in [16, 64, 256]:
        n = int(np.log2(N))
        frozen = np.zeros(N, dtype=bool)
        frozen[np.random.choice(N, N // 2, replace=False)] = True
        u = np.zeros(N, dtype=int)
        u[~frozen] = np.random.randint(0, 2, N // 2)
        llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
        assert np.array_equal(sc_decode(llr, frozen), u)
        print(f"N={N} SC OK")
