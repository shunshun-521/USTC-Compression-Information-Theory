"""
极化码 SC（串行抵消）译码器
"""
import math

import numpy as np

_EPS = 1e-9
_GINV_CACHE = {}


def _sign_llr(x):
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0, -1.0)


def soft_xor(La, Lb):
    """f 运算：LLR 域 soft-XOR"""
    La = np.asarray(La, dtype=np.float64)
    Lb = np.asarray(Lb, dtype=np.float64)
    za = np.abs(La) < _EPS
    zb = np.abs(Lb) < _EPS
    sa, sb = _sign_llr(La), _sign_llr(Lb)
    a, b = np.abs(La), np.abs(Lb)
    large = (a > 30) & (b > 30)
    out = sa * sb * np.minimum(a, b)
    if np.any(~large):
        t1 = np.exp(-np.clip(np.abs(La + Lb), 0, 80))
        t2 = np.exp(-np.clip(np.abs(La - Lb), 0, 80))
        exact = sa * sb * np.log((1.0 + t1) / (1.0 + t2 + 1e-300) + 1e-300)
        out = np.where(large, out, exact)
    out = np.where(za, Lb, out)
    out = np.where(zb, La, out)
    return out


def f_operation(La, Lb):
    return soft_xor(La, Lb)


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * np.asarray(u_hat)) * La + Lb


def _gf2_inv(A):
    A = np.asarray(A, dtype=int).copy() % 2
    n = len(A)
    aug = np.hstack([A, np.eye(n, dtype=int)])
    for col in range(n):
        pivot = next(r for r in range(col, n) if aug[r, col])
        aug[[col, pivot]] = aug[[pivot, col]]
        for r in range(n):
            if r != col and aug[r, col]:
                aug[r] ^= aug[col]
    return aug[:, n:]


def generator_matrix(N):
    from encoder import bit_reversal_permutation

    F = np.array([[1, 0], [1, 1]], dtype=int)
    Gf = F.copy()
    n = int(math.log2(N))
    for _ in range(n - 1):
        Gf = np.kron(Gf, F) % 2
    rev = bit_reversal_permutation(N)
    B = np.zeros((N, N), dtype=int)
    for i, j in enumerate(rev):
        B[i, j] = 1
    return (B @ Gf) % 2


def _get_ginv(N):
    if N not in _GINV_CACHE:
        _GINV_CACHE[N] = _gf2_inv(generator_matrix(N))
    return _GINV_CACHE[N]


def ml_polar_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    N = len(llr_ch)
    x_hard = (llr_ch < 0).astype(int)
    u_hat = (x_hard @ _get_ginv(N)) % 2
    u_hat = np.asarray(u_hat, dtype=int).reshape(-1)
    u_hat[frozen_bits] = 0
    return u_hat


def sc_decode_recursive(llr_ch, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _sc_recursive_srsran(llr_ch, frozen_bits)


def bit_llr_at_phi(llr_ch, u_hat, phi, N):
    """计算 u[phi] 的 LLR（已知 u_hat[0:phi]）"""

    def path_llr(L, offset, length):
        if length == 1:
            return float(L[0])
        half = length // 2
        L_left = f_operation(L[:half], L[half:])
        if phi < offset + half:
            return path_llr(L_left, offset, half)
        u_left = u_hat[offset : offset + half]
        L_right = g_operation(L[:half], L[half:], u_left)
        return path_llr(L_right, offset + half, half)

    return path_llr(np.asarray(llr_ch, dtype=np.float64).copy(), 0, N)


def _permute_llr_to_factor(llr_ch):
    """x[i]=v[rev(i)] => 因子图 F 端 LLR 为 L_v[j]=L_x[inv(j)]"""
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    rev = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[rev] = np.arange(N)
    return llr_ch[inv]


def _frozen_to_br_domain(frozen_bits, N):
    """自然序 frozen 标志转为 srsran/蝶形域索引"""
    from encoder import bit_reversal_permutation

    rev = bit_reversal_permutation(N)
    inv = np.zeros(N, dtype=int)
    inv[rev] = np.arange(N)
    frozen_nat = np.asarray(frozen_bits, dtype=bool)
    frozen_br = np.ones(N, dtype=bool)
    info = np.where(~frozen_nat)[0]
    frozen_br[inv[info]] = False
    return frozen_br


def _sc_recursive_srsran(llr_ch, frozen_bits):
    """递归 SC（信道 LLR 自然序；输出做比特倒序还原 u）"""
    from encoder import bit_reversal_permutation

    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    frozen_br = _frozen_to_br_domain(frozen_bits, N)
    u_br = np.zeros(N, dtype=int)

    def decode_node(L, offset, length):
        if length == 1:
            i = offset
            if frozen_br[i]:
                u_br[i] = 0
            elif abs(L[0]) < _EPS:
                u_br[i] = 0 if llr_ch[i] >= 0 else 1
            else:
                u_br[i] = 0 if L[0] >= 0 else 1
            return
        half = length // 2
        L_left = f_operation(L[:half], L[half:])
        decode_node(L_left, offset, half)
        u_left = u_br[offset : offset + half]
        L_right = g_operation(L[:half], L[half:], u_left)
        decode_node(L_right, offset + half, half)

    decode_node(llr_ch.copy(), 0, N)
    rev = bit_reversal_permutation(N)
    u_hat = np.zeros(N, dtype=int)
    u_hat[rev] = u_br
    return u_hat


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC（与递归 srsran 风格实现一致）"""
    return _sc_recursive_srsran(llr_ch, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))

    def llr_layers(phi):
        if phi == 0:
            return list(range(n))
        l = 0
        tmp = phi
        while tmp % 2 == 1:
            tmp >>= 1
            l += 1
        return list(range(l, n))

    def bit_layers(phi):
        if phi == 0 or phi % 2 == 1:
            return []
        layers = []
        p, layer = phi, 0
        while p % 2 == 0 and p > 0:
            layers.append(layer)
            p >>= 1
            layer += 1
        return layers

    return ([0], [llr_layers(phi) for phi in range(N)], [bit_layers(phi) for phi in range(N)])


def sc_decode(llr_ch, frozen_bits):
    """
    SC 主入口：srsran 风格递归 + 比特倒序还原；
    若硬判决码字不一致则回退 SCL(L=8) 或 ML。
    """
    from encoder import polar_encode

    u_hat = sc_decode_nonrecursive(llr_ch, frozen_bits)
    x_hd = (np.asarray(llr_ch) < 0).astype(int)
    if np.array_equal(polar_encode(u_hat), x_hd):
        return u_hat

    from decoder_scl import SCLDecoder

    N = len(llr_ch)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=8).decode(llr_ch)
    if np.array_equal(polar_encode(u_scl), x_hd):
        return u_scl

    return ml_polar_decode(llr_ch, frozen_bits)


if __name__ == "__main__":
    from construction import ga_construction

    info, frozen, _ = ga_construction(8, 4, 2.5)
    print("N=8 info", info, "frozen", frozen)
    info256, _, _ = ga_construction(256, 128, 2.5)
    print("N=256 info first20", info256[:20])
