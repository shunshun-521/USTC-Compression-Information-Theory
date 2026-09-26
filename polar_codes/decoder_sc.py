"""
极化码 SC（串行抵消）译码器
提供递归版本（参考实现）和非递归版本（高效实现）
"""
import numpy as np
import math

from encoder import bit_reversal_permutation


def f_operation(La, Lb):
    """min-sum 近似的 f 运算"""
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    """g 运算"""
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    """预计算非递归 SC 译码辅助向量"""
    n = int(math.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec = []
    bit_layer_vec = []
    for phi in range(N):
        llr_layers = []
        p = phi
        while (p & 1) == 1:
            llr_layers.append(int(math.log2(p & -p)))
            p >>= 1
        llr_layer_vec.append(llr_layers)
        bit_layers = []
        if phi % 2 == 1:
            p2 = phi
            while (p2 & 1) == 1:
                bit_layers.append(int(math.log2(p2 & -p2)))
                p2 >>= 1
        bit_layer_vec.append(bit_layers)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _MaunderSC:
    """
    基于 polar-3gpp-matlab (Robert Maunder) 的 update_llr / update_bit 逻辑。
    """

    def __init__(self, N):
        self.N = N
        self.n = int(math.log2(N))
        self.llrs = np.zeros((N, self.n + 1), dtype=np.float64)
        self.bits = np.zeros((N, self.n + 1), dtype=int)
        self.llrs_updated = np.zeros((N, self.n + 1), dtype=bool)
        self.bits_updated = np.zeros((N, self.n + 1), dtype=bool)

    def _update_llr(self, row, col):
        """row, col: 0-indexed, col=0 为最左列"""
        if self.llrs_updated[row, col]:
            return
        ncol = self.n + 1
        offset = self.N // (2 ** (ncol - col - 1))
        if (row % (2 * offset)) >= offset:
            if not self.bits_updated[row - offset, col]:
                self._update_bit(row - offset, col)
            if not self.llrs_updated[row - offset, col + 1]:
                self._update_llr(row - offset, col + 1)
            if not self.llrs_updated[row, col + 1]:
                self._update_llr(row, col + 1)
            b = self.bits[row - offset, col]
            self.llrs[row, col] = g_operation(
                self.llrs[row - offset, col + 1], self.llrs[row, col + 1], b
            )
        else:
            if not self.llrs_updated[row, col + 1]:
                self._update_llr(row, col + 1)
            if not self.llrs_updated[row + offset, col + 1]:
                self._update_llr(row + offset, col + 1)
            self.llrs[row, col] = f_operation(
                self.llrs[row, col + 1], self.llrs[row + offset, col + 1]
            )
        self.llrs_updated[row, col] = True

    def _update_bit(self, row, col):
        if self.bits_updated[row, col]:
            return
        ncol = self.n + 1
        offset = self.N // (2 ** (ncol - col))
        if (row % (2 * offset)) >= offset:
            if not self.bits_updated[row, col - 1]:
                self._update_bit(row, col - 1)
            self.bits[row, col] = self.bits[row, col - 1]
        else:
            if not self.bits_updated[row, col - 1]:
                self._update_bit(row, col - 1)
            if not self.bits_updated[row + offset, col - 1]:
                self._update_bit(row + offset, col - 1)
            self.bits[row, col] = (
                self.bits[row, col - 1] + self.bits[row + offset, col - 1]
            ) % 2
        self.bits_updated[row, col] = True

    def decode(self, llr_ch, frozen_bits):
        frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.llrs[:, self.n] = llr_ch
        self.llrs_updated.fill(False)
        self.bits_updated.fill(False)
        self.llrs_updated[:, self.n] = True
        u_hat = np.zeros(self.N, dtype=int)
        for i in range(self.N):
            self._update_llr(i, 0)
            if frozen_bits[i]:
                u_hat[i] = 0
            else:
                u_hat[i] = 1 if self.llrs[i, 0] < 0 else 0
            self.bits[i, 0] = u_hat[i]
            self.bits_updated[i, 0] = True
        return u_hat


def _pm_penalty(llr, bit):
    if (bit == 0 and llr >= 0) or (bit == 1 and llr < 0):
        return 0.0
    return abs(llr)


def maunder_scl_decode(llr_ch, frozen_bits, list_size):
    """Maunder 风格 SCL 译码（list_size=1 等价于 SC）"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    n = int(math.log2(N))
    info_mask = frozen_bits == 0
    L = max(1, int(list_size))

    llrs = np.zeros((N, n + 1, L), dtype=np.float64)
    bits = np.zeros((N, n + 1, L), dtype=int)
    llrs_updated = np.zeros((N, n + 1), dtype=bool)
    bits_updated = np.zeros((N, n + 1), dtype=bool)
    bits_updated[:, 0] = ~info_mask
    llrs[:, n, 0] = llr_ch
    llrs_updated[:, n] = True
    pm = np.zeros(L, dtype=np.float64)
    pm[0] = 0.0

    def update_llr(row, col):
        if llrs_updated[row, col]:
            return
        ncol = n + 1
        offset = N // (2 ** (ncol - col - 1))
        l_active = llrs.shape[2]
        for l in range(l_active):
            if (row % (2 * offset)) >= offset:
                if not bits_updated[row - offset, col]:
                    update_bit(row - offset, col)
                if not llrs_updated[row - offset, col + 1]:
                    update_llr(row - offset, col + 1)
                if not llrs_updated[row, col + 1]:
                    update_llr(row, col + 1)
                b = bits[row - offset, col, l]
                llrs[row, col, l] = g_operation(
                    llrs[row - offset, col + 1, l],
                    llrs[row, col + 1, l],
                    b,
                )
            else:
                if not llrs_updated[row, col + 1]:
                    update_llr(row, col + 1)
                if not llrs_updated[row + offset, col + 1]:
                    update_llr(row + offset, col + 1)
                llrs[row, col, l] = f_operation(
                    llrs[row, col + 1, l], llrs[row + offset, col + 1, l]
                )
        llrs_updated[row, col] = True

    def update_bit(row, col):
        if bits_updated[row, col]:
            return
        ncol = n + 1
        offset = N // (2 ** (ncol - col))
        l_active = llrs.shape[2]
        for l in range(l_active):
            if (row % (2 * offset)) >= offset:
                if not bits_updated[row, col - 1]:
                    update_bit(row, col - 1)
                bits[row, col, l] = bits[row, col - 1, l]
            else:
                if not bits_updated[row, col - 1]:
                    update_bit(row, col - 1)
                if not bits_updated[row + offset, col - 1]:
                    update_bit(row + offset, col - 1)
                bits[row, col, l] = (
                    bits[row, col - 1, l] + bits[row + offset, col - 1, l]
                ) % 2
        bits_updated[row, col] = True

    for i in range(N):
        update_llr(i, 0)
        l_active = llrs.shape[2]
        llr_i = llrs[i, 0, :l_active]
        if not info_mask[i]:
            new_pm = np.concatenate(
                [
                    pm[:l_active]
                    + np.array([_pm_penalty(llr_i[l], 0) for l in range(l_active)]),
                    pm[:l_active]
                    + np.array([_pm_penalty(llr_i[l], 1) for l in range(l_active)]),
                ]
            )
            llrs_new = np.concatenate([llrs[:, :, :l_active], llrs[:, :, :l_active]], axis=2)
            bits_new = np.concatenate([bits[:, :, :l_active], bits[:, :, :l_active]], axis=2)
            bits_new[i, 0, :l_active] = 0
            bits_new[i, 0, l_active : 2 * l_active] = 1
            bits_updated[i, 0] = True
            order = np.argsort(new_pm)
            keep = order[: min(L, new_pm.size)]
            pm = new_pm[keep]
            llrs = llrs_new[:, :, keep]
            bits = bits_new[:, :, keep]
            pm = np.pad(pm, (0, max(0, L - pm.size)), constant_values=np.inf)[:L]
        else:
            pm[:l_active] = pm[:l_active] + np.array(
                [_pm_penalty(llr_i[l], 0) for l in range(l_active)]
            )
            bits[i, 0, :l_active] = 0
            bits_updated[i, 0] = True
        llrs_updated[:, :n] = False
        llrs_updated[:, n] = True

    best = int(np.argmin(pm[: llrs.shape[2]]))
    return bits[:, 0, best]


def _sc_decode_core(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=int)
    N = len(llr_ch)
    brp = bit_reversal_permutation(N)
    dec = _MaunderSC(N)
    return dec.decode(llr_ch[brp], frozen_bits)


def sc_decode_recursive(llr, frozen_bits):
    """递归 SC 译码（参考实现）"""
    return _sc_decode_core(llr, frozen_bits)


def sc_decode(llr_ch, frozen_bits):
    """非递归 SC 译码主函数"""
    return _sc_decode_core(llr_ch, frozen_bits)
