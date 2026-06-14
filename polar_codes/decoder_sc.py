"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


# ==================== 基本运算 ====================


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    支持向量化（La, Lb 为同形状 numpy 数组）
    """
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    sa = np.sign(La).copy()
    sb = np.sign(Lb).copy()
    if sa.ndim == 0:
        if sa == 0:
            sa = np.float64(1)
        if sb == 0:
            sb = np.float64(1)
        return sa * sb * min(abs(La), abs(Lb))
    sa[sa == 0] = 1
    sb[sb == 0] = 1
    return sa * sb * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """
    g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb
    """
    return (1 - 2 * u_hat) * La + Lb


def _bit_reversed(i, n):
    return int(f'{i:0{n}b}'[::-1], 2)


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


# ==================== 递归 SC 译码（参考实现）====================


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（按比特倒序逐步递归，与 sc_decode 等价）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr
    u_hat = np.zeros(N, dtype=int)
    decode_order = [_bit_reversed(i, n) for i in range(N)]

    def decode_step(step):
        if step >= N:
            return
        phi = decode_order[step]
        for s in range(n - _active_llr_level(phi, n), n):
            block = 2 ** (s + 1)
            branch = block // 2
            for j in range(phi, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )
        if frozen_bits[phi]:
            B[phi, n] = 0
        else:
            B[phi, n] = 0 if L[phi, n] >= 0 else 1
        u_hat[phi] = B[phi, n]
        if phi >= N // 2:
            for s in range(n, n - _active_bit_level(phi, n), -1):
                block = 2 ** s
                branch = block // 2
                for j in range(phi, -1, -block):
                    if j % block >= branch:
                        B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                        B[j, s - 1] = B[j, s]
        decode_step(step + 1)

    decode_step(0)
    return u_hat


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的辅助向量。
    返回比特倒序译码顺序及每步活跃层信息。
    """
    n = int(np.log2(N))
    lambda_offset = [1 << layer for layer in range(n + 1)]
    decode_order = [_bit_reversed(i, n) for i in range(N)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in decode_order:
        llr_layer_vec.append(list(range(n - _active_llr_level(phi, n), n)))
        if phi >= N // 2:
            bit_layer_vec.append(list(range(n, n - _active_bit_level(phi, n), -1)))
        else:
            bit_layer_vec.append([])
    return lambda_offset, llr_layer_vec, bit_layer_vec, decode_order


_SC_CACHE = {}


def _get_sc_cache(N):
    if N not in _SC_CACHE:
        _SC_CACHE[N] = precompute_sc_indices(N)
    return _SC_CACHE[N]


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（比特倒序、分层 LLR 更新）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    _, llr_layer_vec, bit_layer_vec, decode_order = _get_sc_cache(N)

    L = np.full((N, n + 1), np.nan, dtype=np.float64)
    B = np.zeros((N, n + 1), dtype=int)
    L[:, 0] = llr_ch
    u_hat = np.zeros(N, dtype=int)

    for idx, phi in enumerate(decode_order):
        for s in llr_layer_vec[idx]:
            block = 2 ** (s + 1)
            branch = block // 2
            for j in range(phi, N, block):
                if j % block < branch:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch, s], L[j, s], B[j - branch, s + 1]
                    )

        if frozen_bits[phi]:
            B[phi, n] = 0
        else:
            B[phi, n] = 0 if L[phi, n] >= 0 else 1
        u_hat[phi] = B[phi, n]

        for s in bit_layer_vec[idx]:
            block = 2 ** s
            branch = block // 2
            for j in range(phi, -1, -block):
                if j % block >= branch:
                    B[j - branch, s - 1] = B[j, s] ^ B[j - branch, s]
                    B[j, s - 1] = B[j, s]

    return u_hat
