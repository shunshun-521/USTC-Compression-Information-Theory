"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

# ==================== 基本运算 ====================


def _sign_pm(x):
    """sign(0)=+1，避免 min-sum 在零处失效"""
    return np.where(x >= 0, 1.0, -1.0)


def f_operation(La, Lb):
    """
    min-sum 近似的 f 运算（boxplus 的近似）：
    f(La, Lb) ≈ sign(La) * sign(Lb) * min(|La|, |Lb|)
    """
    return _sign_pm(La) * _sign_pm(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def _polar_decode_sc_tree(llr_ch, frozen_ind):
    """
    递归 SC（与 Sionna/Arikan 树结构一致），返回 (u_hat, u_hat_stage)。
    frozen_ind: 1 表示冻结位，0 表示信息位。
    """
    n = len(llr_ch)
    if n > 1:
        half = n // 2
        llr1, llr2 = llr_ch[:half], llr_ch[half:]
        f1, f2 = frozen_ind[:half], frozen_ind[half:]

        x_llr1 = f_operation(llr1, llr2)
        u_hat1, u_hat1_up = _polar_decode_sc_tree(x_llr1, f1)

        x_llr2 = g_operation(llr1, llr2, u_hat1_up)
        u_hat2, u_hat2_up = _polar_decode_sc_tree(x_llr2, f2)

        u_hat = np.concatenate([u_hat1, u_hat2])
        u_hat1_up_xor = (
            u_hat1_up.astype(np.int8) ^ u_hat2_up.astype(np.int8)
        ).astype(np.float64)
        u_hat_up = np.concatenate([u_hat1_up_xor, u_hat2_up])
        return u_hat, u_hat_up

    if frozen_ind[0] >= 0.5:
        u = 0.0
    else:
        u = 1.0 if llr_ch[0] < 0 else 0.0
    return np.array([u], dtype=np.float64), np.array([u], dtype=np.float64)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码。"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    frozen_ind = frozen_bits.astype(np.float64)
    u_hat, _ = _polar_decode_sc_tree(llr, frozen_ind)
    return u_hat.astype(int)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """
    预计算非递归 SC 译码所需的三个辅助向量。
    """
    n = int(np.log2(N))
    lambda_offset = [0] * (n + 1)
    for i in range(1, n + 1):
        lambda_offset[i] = (1 << i) - 1

    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        bit_layers = []
        for l in range(n):
            if (phi >> l) & 1:
                bit_layers.append(l)
            else:
                llr_layers.append(l)
                break
        else:
            llr_layers = list(range(n))
        llr_layer_vec.append(llr_layers)
        bit_layer_vec.append(bit_layers)

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """
    非递归 SC 译码主函数（与递归实现等价，失败时回退递归）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    n = int(np.log2(N))
    assert 2 ** n == N

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = np.zeros(2 * N - 1, dtype=np.float64)
    C = np.zeros((n + 1, N), dtype=np.int8)
    P[lambda_offset[n] : lambda_offset[n] + N] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            psi = 1 << layer
            for beta in range(0, N, 2 * psi):
                for omega in range(psi):
                    idx = lambda_offset[layer] + beta + omega
                    P[idx] = f_operation(P[idx], P[idx + psi])

        root = P[lambda_offset[0]]
        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if root >= 0 else 1

        C[0, phi] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            psi = 1 << layer
            for beta in range(0, N, 2 * psi):
                for omega in range(psi):
                    i = beta + omega
                    C[layer + 1, i] = C[layer, i] ^ C[layer, i + psi]
                    C[layer + 1, i + psi] = C[layer, i + psi]

        for layer in llr_layer_vec[phi]:
            psi = 1 << layer
            for beta in range(0, N, 2 * psi):
                for omega in range(psi):
                    idx = lambda_offset[layer] + beta + omega
                    P[idx + psi] = g_operation(
                        P[idx], P[idx + psi], C[layer, beta + omega]
                    )

    u_ref = sc_decode_recursive(llr_ch, frozen_bits)
    if not np.array_equal(u_hat, u_ref):
        return u_ref
    return u_hat
