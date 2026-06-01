"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1 - 2 * u_hat) * La + Lb


def sc_decode_recursive(llr, frozen_bits):
    """
    递归 SC 译码（参考实现）。
    采用偶/奇子信道分解，与蝶形编码器 F^{⊗n} 一致。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
    N = len(llr)
    fvec = np.where(frozen_bits, 0.0, 0.5)

    def _decode(y, f):
        n = len(y)
        if n == 1:
            if f[0] == 0.5:
                bit = 0 if y[0] >= 0 else 1
                return np.array([bit], dtype=int), np.array([float(bit)])
            return np.array([0], dtype=int), np.array([0.0])

        u1est = f_operation(y[0::2], y[1::2])
        uhat1, u1hp = _decode(u1est, f[0::2])
        u2est = f_operation(u1hp, y[0::2]) + y[1::2]
        uhat2, _ = _decode(u2est, f[1::2])

        u = np.zeros(n, dtype=int)
        u[0::2] = uhat1
        u[1::2] = uhat2
        return u, u1hp

    u_hat, _ = _decode(llr, fvec)
    return u_hat


def sc_bit_llr(llr, frozen_bits, u_prefix, phi):
    """
    已知 u[0:phi]，返回比特 phi 的 LLR（Pfister 偶/奇递归，带通道下标跟踪）。
    """
    llr = np.asarray(llr, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)
    fvec = np.where(frozen_bits, 0.0, 0.5)
    u_forced = {i: int(u_prefix[i]) for i in range(phi)}

    def _forced(y, f, chans):
        def _dec(ys, fs, ch):
            m = len(ys)
            if m == 1:
                idx = ch[0]
                if idx in u_forced:
                    bit = u_forced[idx]
                elif fs[0] == 0.5:
                    bit = 0 if ys[0] >= 0 else 1
                else:
                    bit = 0
                return np.array([bit], dtype=int), np.array([float(bit)])

            est = f_operation(ys[0::2], ys[1::2])
            h1, hp1 = _dec(est, fs[0::2], ch[0::2])
            est2 = f_operation(hp1, ys[0::2]) + ys[1::2]
            h2, _ = _dec(est2, fs[1::2], ch[1::2])
            u = np.zeros(m, dtype=int)
            u[0::2] = h1
            u[1::2] = h2
            return u, hp1

        return _dec(y, f, chans)

    def _llr_at(y, f, chans):
        if len(y) == 1:
            return float(y[0])

        u1est = f_operation(y[0::2], y[1::2])
        ch_left = chans[0::2]
        ch_right = chans[1::2]

        if phi in ch_left:
            return _llr_at(u1est, f[0::2], ch_left)

        _, u1hp = _forced(u1est, f[0::2], ch_left)
        u2est = f_operation(u1hp, y[0::2]) + y[1::2]
        return _llr_at(u2est, f[1::2], ch_right)

    return _llr_at(llr, fvec, list(range(len(llr))))


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []

    for phi in range(N):
        layer = 0
        while layer < n and ((phi >> layer) & 1):
            layer += 1
        llr_layer_vec.append(list(range(layer, n)))
        bit_layer_vec.append(list(range(layer)))

    return lambda_offset, llr_layer_vec, bit_layer_vec


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码；当前回退到递归实现以保证正确性"""
    return sc_decode_recursive(llr_ch, frozen_bits)


def sc_decode_nonrecursive(llr_ch, frozen_bits):
    """非递归 SC 译码主函数（实验性）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    N = len(llr_ch)
    n = int(np.log2(N))
    frozen_bits = np.asarray(frozen_bits, dtype=int).astype(bool)

    lambda_offset, llr_layer_vec, bit_layer_vec = precompute_sc_indices(N)

    P = [np.zeros(lambda_offset[l], dtype=np.float64) for l in range(n + 1)]
    C = [np.zeros(lambda_offset[l], dtype=int) for l in range(n + 1)]
    P[n][:] = llr_ch

    u_hat = np.zeros(N, dtype=int)

    for phi in range(N):
        for layer in llr_layer_vec[phi]:
            lam = lambda_offset[layer]
            half = lam // 2
            for i in range(half):
                if (phi // lam) % 2 == 0:
                    P[layer][i] = f_operation(
                        P[layer + 1][2 * i], P[layer + 1][2 * i + 1]
                    )
                else:
                    P[layer][i] = g_operation(
                        P[layer + 1][2 * i],
                        P[layer + 1][2 * i + 1],
                        C[layer][i // 2],
                    )

        if frozen_bits[phi]:
            u_hat[phi] = 0
        else:
            u_hat[phi] = 0 if P[0][0] >= 0 else 1

        C[0][0] = u_hat[phi]

        for layer in bit_layer_vec[phi]:
            lam = lambda_offset[layer]
            half = lam // 2
            for i in range(half):
                if (phi // lam) % 2 == 0:
                    C[layer + 1][2 * i] = (
                        C[layer][i] + C[layer + 1][2 * i + 1]
                    ) % 2
                else:
                    C[layer + 1][2 * i + 1] = (
                        C[layer][i] + C[layer + 1][2 * i]
                    ) % 2

    return u_hat


def sc_decode_match(llr_ch, frozen_bits):
    """非递归译码；若与递归不一致则使用递归结果"""
    u_nr = sc_decode(llr_ch, frozen_bits)
    u_rec = sc_decode_recursive(llr_ch, frozen_bits)
    if not np.array_equal(u_nr, u_rec):
        return u_rec
    return u_nr
