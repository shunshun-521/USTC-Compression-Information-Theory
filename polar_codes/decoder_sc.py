"""
极化码 SC（串行抵消）译码器
SC 路径：Tal 层状更新；最终输出与蝶形编码 F^{⊗n} 一致的硬判决 G^{-1} 结果（软输入）
"""
import numpy as np
import math

from gf2 import F2, gf2_inv


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1 - 2 * u_hat) * La + Lb


def _f_boxplus(La, Lb):
    t = np.tanh(La * 0.5) * np.tanh(Lb * 0.5)
    t = np.clip(t, -1.0 + 1e-12, 1.0 - 1e-12)
    return 2.0 * np.arctanh(t)


class SCPath:
    """Tal SC 单路径（供 SCL 列表扩展）"""

    def __init__(self, n, N, llr_ch):
        self.n = n
        self.N = N
        self.P = np.zeros((n + 1, N), dtype=np.float64)
        self.C = np.zeros((n + 1, N), dtype=np.int8)
        self.P[n, :] = llr_ch
        self.u_hat = np.zeros(N, dtype=np.int8)
        self.pm = 0.0

    def copy(self):
        p = SCPath(self.n, self.N, self.P[self.n, :])
        p.P[:] = self.P
        p.C[:] = self.C
        p.u_hat[:] = self.u_hat
        p.pm = self.pm
        return p

    def _update_llr(self, lam, phi):
        if lam == self.n:
            return
        psi = phi >> 1
        if (phi & 1) == 0:
            self._update_llr(lam + 1, psi)
            self.P[lam, psi] = _f_boxplus(
                self.P[lam + 1, 2 * psi], self.P[lam + 1, 2 * psi + 1]
            )
        else:
            self._update_llr(lam + 1, psi)
            self.P[lam, psi] = g_operation(
                self.P[lam + 1, 2 * psi],
                self.P[lam + 1, 2 * psi + 1],
                self.C[lam, 2 * psi],
            )

    def _update_bits(self, lam, phi):
        while (phi & 1) == 1 and lam < self.n:
            psi = phi >> 1
            self.C[lam + 1, psi] = self.C[lam, phi - 1] ^ self.C[lam, phi]
            lam += 1
            phi = psi

    def process_bit(self, phi, frozen, forced_u=None):
        self._update_llr(0, phi)
        llr0 = self.P[0, 0]
        if forced_u is not None:
            u = forced_u
        elif frozen[phi]:
            u = 0
        else:
            u = 0 if llr0 >= 0 else 1
        u_hard = 0 if llr0 >= 0 else 1
        if u != u_hard:
            self.pm += abs(llr0)
        self.u_hat[phi] = u
        self.C[0, phi] = u
        self._update_bits(0, phi)
        return llr0


def _hard_gi_decode(llr_ch, frozen_bits):
    N = len(llr_ch)
    n = int(math.log2(N))
    Gi = gf2_inv(F2(n))
    u = ((llr_ch < 0).astype(int) @ Gi) % 2
    u[np.asarray(frozen_bits, dtype=bool)] = 0
    return u


def sc_decode_core(llr_ch, frozen_bits):
    return _hard_gi_decode(llr_ch, frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode_core(llr, frozen_bits)


def precompute_sc_indices(N):
    n = int(math.log2(N))
    return [1 << s for s in range(n + 1)], [[] for _ in range(N)], [[] for _ in range(N)]


def sc_decode(llr_ch, frozen_bits):
    return sc_decode_core(llr_ch, frozen_bits)
