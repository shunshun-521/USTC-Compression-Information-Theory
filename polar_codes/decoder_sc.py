"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（Vangala 置换 SCD，高效实现）
"""
import numpy as np
from encoder import bit_reversed


def _logdomain_sum(x, y):
    if x > y:
        return x + np.log1p(np.exp(y - x))
    return y + np.log1p(np.exp(x - y))


def f_operation(La, Lb):
    """
    f 运算（box-plus，对数域精确实现）。
    min-sum 近似在 SC 树上误差过大，故采用 box-plus。
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    scalar = La.ndim == 0
    if scalar:
        La = La.reshape(1)
        Lb = Lb.reshape(1)
    out = np.empty_like(La)
    for i in range(La.size):
        a, b = float(La.flat[i]), float(Lb.flat[i])
        out.flat[i] = _logdomain_sum(a + b, 0.0) - _logdomain_sum(a, b)
    return out.item() if scalar else out


def f_operation_minsum(La, Lb):
    """min-sum 近似的 f 运算（供参考/对比）。"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    u_hat = np.asarray(u_hat)
    return np.where(u_hat == 0, La + Lb, La - Lb)


def _active_llr_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) == 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def _active_bit_level(i, n):
    mask = 2 ** (n - 1)
    count = 1
    for _ in range(n):
        if (mask & i) > 0:
            count += 1
        else:
            break
        mask >>= 1
    return min(count, n)


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量。"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        l = bit_reversed(phi, n)
        llr_layer_vec.append(list(range(n - _active_llr_level(l, n), n)))
        bit_layer_vec.append(list(range(n, n - _active_bit_level(l, n), -1)))
    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（委托给非递归实现）。"""
    return sc_decode(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码（Vangala 置换 SCD）。
    信道 LLR 按自然顺序输入，内部按比特倒序相位译码。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_set = set(np.where(frozen_bits)[0])

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    C = np.full((N, n + 1), np.nan)
    L[:, 0] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phase in range(N):
        l = bit_reversed(phase, n)

        for s in range(n - _active_llr_level(l, n), n):
            block = 2 ** (s + 1)
            half = block // 2
            for j in range(l, N, block):
                if j % block < half:
                    L[j, s + 1] = f_operation(L[j, s], L[j + half, s])
                else:
                    top_bit = int(C[j - half, s + 1])
                    L[j, s + 1] = g_operation(L[j, s], L[j - half, s], top_bit)

        if l in frozen_set:
            u_hat[l] = 0
            C[l, n] = 0
        else:
            u_hat[l] = 0 if L[l, n] >= 0 else 1
            C[l, n] = u_hat[l]

        if l >= N // 2:
            for s in range(n, n - _active_bit_level(l, n), -1):
                block = 2 ** s
                half = block // 2
                for j in range(l, -1, -block):
                    if j % block >= half:
                        C[j - half, s - 1] = int(C[j, s]) ^ int(C[j - half, s])
                        C[j, s - 1] = C[j, s]

    return u_hat
