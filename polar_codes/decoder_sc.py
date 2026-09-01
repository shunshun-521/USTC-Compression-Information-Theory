"""
极化码 SC（串行抵消）译码器
基于 3GPP polar decoder 因子图（robmaunder/polar-3gpp-matlab）
"""
import numpy as np


def f_operation(La, Lb):
    return np.sign(La) * np.sign(Lb) * np.minimum(np.abs(La), np.abs(Lb))


def g_operation(La, Lb, u_hat):
    return (1.0 - 2.0 * u_hat) * La + Lb


def precompute_sc_indices(N):
    n = int(np.log2(N))
    lambda_offset = [1 << i for i in range(n + 1)]
    llr_layer_vec, bit_layer_vec = [], []
    for phi in range(N):
        layers_llr = []
        for i in range(n):
            if (phi >> i) & 1 == 0:
                layers_llr = list(range(i, n))
                break
        llr_layer_vec.append(layers_llr)
        layers_bit = []
        for i in range(n):
            if (phi >> i) & 1 == 1:
                layers_bit.append(i)
            else:
                break
        bit_layer_vec.append(layers_bit)
    return lambda_offset, llr_layer_vec, bit_layer_vec


class _SCState:
    """MATLAB 1-indexed 内部表示"""

    def __init__(self, llr_ch, frozen_bits):
        self.N = len(llr_ch)
        self.n = int(np.log2(self.N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)

        self.bits = np.zeros((self.N + 1, self.n + 2), dtype=np.int8)
        self.bits_updated = np.zeros((self.N + 1, self.n + 2), dtype=bool)
        self.bits_updated[1:self.N + 1, 1] = self.frozen_bits

        self.llrs = np.zeros((self.N + 1, self.n + 2))
        self.llrs[1:self.N + 1, self.n + 1] = llr_ch
        self.llrs_updated = np.zeros((self.N + 1, self.n + 2), dtype=bool)
        self.llrs_updated[1:self.N + 1, self.n + 1] = True

    def _update_bit(self, row, col):
        if self.bits_updated[row, col]:
            return
        offset = max(1, self.N // (2 ** (self.n + 2 - col)))
        for l in range(1):
            if (row - 1) % (2 * offset) >= offset:
                if not self.bits_updated[row, col - 1]:
                    self._update_bit(row, col - 1)
                self.bits[row, col] = self.bits[row, col - 1]
            else:
                if not self.bits_updated[row, col - 1]:
                    self._update_bit(row, col - 1)
                if not self.bits_updated[row + offset, col - 1]:
                    self._update_bit(row + offset, col - 1)
                self.bits[row, col] = (
                    self.bits[row, col - 1] ^ self.bits[row + offset, col - 1]
                )
        self.bits_updated[row, col] = True

    def _update_llr(self, row, col):
        if col > self.n + 1 or self.llrs_updated[row, col]:
            return
        offset = max(1, self.N // (2 ** (self.n + 1 - col)))
        if (row - 1) % (2 * offset) >= offset:
            if not self.bits_updated[row - offset, col]:
                self._update_bit(row - offset, col)
            if not self.llrs_updated[row - offset, col + 1]:
                self._update_llr(row - offset, col + 1)
            if not self.llrs_updated[row, col + 1]:
                self._update_llr(row, col + 1)
            u = self.bits[row - offset, col]
            self.llrs[row, col] = g_operation(
                self.llrs[row - offset, col + 1],
                self.llrs[row, col + 1],
                u
            )
        else:
            if not self.llrs_updated[row, col + 1]:
                self._update_llr(row, col + 1)
            if not self.llrs_updated[row + offset, col + 1]:
                self._update_llr(row + offset, col + 1)
            self.llrs[row, col] = f_operation(
                self.llrs[row, col + 1],
                self.llrs[row + offset, col + 1]
            )
        self.llrs_updated[row, col] = True


def sc_decode(llr_ch, frozen_bits):
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    frozen_bits = np.asarray(frozen_bits, dtype=bool)
    state = _SCState(llr_ch, frozen_bits)
    u_hat = np.zeros(state.N, dtype=int)

    for i in range(1, state.N + 1):
        state._update_llr(i, 1)
        if frozen_bits[i - 1]:
            u_hat[i - 1] = 0
        else:
            u_hat[i - 1] = 0 if state.llrs[i, 1] >= 0 else 1
        state.bits[i, 1] = u_hat[i - 1]
        state.bits_updated[i, 1] = True

    return u_hat


def sc_decode_recursive(llr, frozen_bits):
    return sc_decode(llr, frozen_bits)
