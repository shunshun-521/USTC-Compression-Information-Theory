"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np

from encoder import bit_reversal_permutation

# ==================== 基本运算 ====================


def f_boxplus(La, Lb):
    """boxplus（f 运算，精确 LLR 合并）"""
    La = np.clip(La, -30.0, 30.0)
    Lb = np.clip(Lb, -30.0, 30.0)
    return np.log1p(np.exp(La + Lb)) - np.log(np.exp(La) + np.exp(Lb))


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算：g(La, Lb, u_hat) = (1 - 2*u_hat) * La + Lb"""
    return (1 - 2 * u_hat) * La + Lb


# ==================== 递归 SC 译码（参考实现）====================


def _sc_rec_sionna(llr, frozen_ind, use_boxplus=True):
    """
    递归 SC（与极化码标准因子图一致）。
    frozen_ind: 1=冻结，0=信息；g 运算使用部分和 u_hat_up。
    """
    n = len(llr)
    f_fn = f_boxplus if use_boxplus else f_operation

    if n == 1:
        if frozen_ind[0] == 1:
            u = np.array([0.0])
        else:
            u = np.array([0.0 if llr[0] >= 0 else 1.0])
        return u, u.copy()

    half = n // 2
    llr1 = llr[:half]
    llr2 = llr[half:]
    fi1 = frozen_ind[:half]
    fi2 = frozen_ind[half:]

    llr_u = f_fn(llr1, llr2)
    u1, u1_up = _sc_rec_sionna(llr_u, fi1, use_boxplus)
    llr_d = g_operation(llr1, llr2, u1_up)
    u2, u2_up = _sc_rec_sionna(llr_d, fi2, use_boxplus)

    u_hat = np.concatenate([u1, u2])
    u1_up_int = (u1_up.astype(int) ^ u2_up.astype(int)).astype(float)
    u_hat_up = np.concatenate([u1_up_int, u2_up])
    return u_hat, u_hat_up


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（信道 LLR 已为比特倒序顺序）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_ind = np.asarray(frozen_bits, dtype=float)
    if frozen_ind.dtype == bool:
        frozen_ind = frozen_ind.astype(float)
    u_hat, _ = _sc_rec_sionna(llr, frozen_ind, use_boxplus=True)
    return u_hat.astype(int)


# ==================== 非递归 SC 译码（高效实现）====================


def precompute_sc_indices(N):
    """预计算非递归 SC 译码所需的辅助向量"""
    n = int(np.log2(N))
    lambda_offset = list(range(N))
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        v = phi
        l = 0
        while v % 2 == 1:
            v //= 2
            l += 1
        llr_layer_vec.append(list(range(l, n)))

        v = (phi + 1) // 2
        l = 0
        while v % 2 == 1:
            v //= 2
            l += 1
        bit_layer_vec.append(list(range(l, n)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode_nonrecursive(llr, frozen_bits):
    """非递归 SC（min-sum f，与递归结果一致）"""
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr)
    n = int(np.log2(N))
    m = n + 1
    L = np.zeros((m, N), dtype=np.float64)
    C = np.zeros((m, N), dtype=np.int8)
    L[m - 1, :] = llr

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)
    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            span = 1 << layer
            for i in range(0, N, 2 * span):
                La = L[layer + 1, i : i + span]
                Lb = L[layer + 1, i + span : i + 2 * span]
                L[layer, i : i + span] = f_operation(La, Lb)

        for layer in reversed(llr_layer_vec[phi]):
            span = 1 << layer
            for i in range(0, N, 2 * span):
                La = L[layer + 1, i : i + span]
                Lb = L[layer + 1, i + span : i + 2 * span]
                u_part = C[layer + 1, i : i + span]
                L[layer, i : i + span] = g_operation(La, Lb, u_part)

        if frozen_bits[phi]:
            u_bit = 0
        else:
            u_bit = 0 if L[0, 0] >= 0 else 1
        u_hat[phi] = u_bit
        C[0, 0] = u_bit

        for layer in bit_layer_vec[phi]:
            span = 1 << layer
            for i in range(0, N, 2 * span):
                C[layer + 1, i + span : i + 2 * span] = (
                    C[layer, i : i + span] + C[layer + 1, i + span : i + 2 * span]
                ) % 2

    return u_hat


def sc_decode(llr_ch, frozen_bits):
    """
    SC 译码入口：对信道 LLR 做比特倒序后译码（与蝶形+BR 编码器配套）。
    """
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    br = bit_reversal_permutation(len(llr_ch))
    return sc_decode_recursive(llr_ch[br], frozen_bits)


def sc_decode_verify(llr_ch, frozen_bits):
    """递归与非递归结果一致性检查"""
    br = bit_reversal_permutation(len(llr_ch))
    llr = llr_ch[br]
    a = sc_decode_recursive(llr, frozen_bits)
    b = sc_decode_nonrecursive(llr, frozen_bits)
    return a, b, np.array_equal(a, b)
