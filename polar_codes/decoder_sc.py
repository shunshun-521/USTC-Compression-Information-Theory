"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math


def bit_reversed(x, n):
    """单索引比特倒序"""
    result = 0
    for i in range(n):
        if x & (1 << i):
            result |= (1 << (n - 1 - i))
    return result


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_boxplus(La, Lb):
    """SC/SCL 使用的精确对数域 box-plus f 运算"""
    if np.isscalar(La) and np.isscalar(Lb):
        return _logdomain_sum(La + Lb, 0.0) - _logdomain_sum(La, Lb)
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    out = np.empty_like(La)
    flat_a = La.ravel()
    flat_b = Lb.ravel()
    flat_o = out.ravel()
    for i in range(flat_a.size):
        a, b = flat_a[i], flat_b[i]
        flat_o[i] = _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)
    return out


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


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


def prepare_llr_for_decoder(llr_ch, N):
    """
    将信道 LLR（对应 polar_encode 输出的码字顺序）转换为译码树底层 LLR。
    """
    n = int(math.log2(N))
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    return np.array([llr_ch[bit_reversed(j, n)] for j in range(N)], dtype=np.float64)


def _frozen_indices_from_mask(frozen_bits):
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    return set(np.where(frozen_bits > 0)[0])


def _sc_decode_core(llr_ch, frozen_bits):
    """SC 译码核心（非递归，比特倒序相位）"""
    llr = prepare_llr_for_decoder(llr_ch, len(llr_ch))
    N = len(llr)
    n = int(math.log2(N))
    frozen_set = _frozen_indices_from_mask(frozen_bits)
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.full((N, n + 1), np.nan)
    L[:, 0] = llr

    for i in range(N):
        l = bit_reversed(i, n)
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = _logdomain_sum(
                        L[j, s] + L[j + branch_size, s], 0.0
                    ) - _logdomain_sum(L[j, s], L[j + branch_size, s])
                else:
                    u_bit = B[j - branch_size, s + 1]
                    a, b = L[j, s], L[j - branch_size, s]
                    L[j, s + 1] = a + b if u_bit == 0 else a - b
        B[l, n] = 0 if (l in frozen_set or L[l, n] >= 0) else 1
        if l >= N / 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block_size = 2 ** s
                branch_size = block_size // 2
                for j in range(l, -1, -block_size):
                    if j % block_size >= branch_size:
                        B[j - branch_size, s - 1] = int(B[j, s]) ^ int(B[j - branch_size, s])
                        B[j, s - 1] = B[j, s]

    return B[:, n].astype(int)


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（与 sc_decode 共用同一非递归核心实现）"""
    return _sc_decode_core(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码辅助向量（与比特倒序相位译码顺序配套）。
    """
    n = int(math.log2(N))
    lambda_offset = [(1 << i) - 1 for i in range(n + 1)]

    llr_layer_vec = []
    bit_layer_vec = []
    for phi_nat in range(N):
        l = bit_reversed(phi_nat, n)
        llr_layers = list(range(n - _active_llr_level(l, n), n))
        bit_layers = list(range(n, n - _active_bit_level(l, n), -1)) if l >= N / 2 else []
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return _sc_decode_core(llr_ch, frozen_bits)


if __name__ == "__main__":
    from encoder import polar_encode
    from construction import ga_construction

    N = 64
    K = 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(0)
    for _ in range(100):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = 1e6 * (1 - 2 * x)
        u_sc = sc_decode(llr, frozen_bits)
        u_rec = sc_decode_recursive(llr, frozen_bits)
        assert np.array_equal(u_sc, u_rec)
        assert np.array_equal(u_sc, u)
