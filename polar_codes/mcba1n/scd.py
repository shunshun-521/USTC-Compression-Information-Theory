import numpy as np

from mcba1n.decoder_utils import active_bit_level, active_llr_level, hard_decision, lower_llr, upper_llr
from mcba1n.utils import bit_reversed


class SCD:
    def __init__(self, llr_ch, frozen_indices, N, n):
        self.N = N
        self.n = n
        self.frozen = frozen_indices
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch

    def decode(self):
        for l in [bit_reversed(i, self.n) for i in range(self.N)]:
            self.update_llrs(l)
            if l in self.frozen:
                self.B[l, self.n] = 0
            else:
                self.B[l, self.n] = hard_decision(self.L[l, self.n])
            self.update_bits(l)
        return self.B[:, self.n].astype(int)

    def update_llrs(self, l):
        for s in range(self.n - active_llr_level(l, self.n), self.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    self.L[j, s + 1] = upper_llr(self.L[j, s], self.L[j + branch_size, s])
                else:
                    self.L[j, s + 1] = lower_llr(
                        self.L[j - branch_size, s],
                        self.L[j, s],
                        self.B[j - branch_size, s + 1],
                    )

    def update_bits(self, l):
        if l < self.N / 2:
            return
        for s in range(self.n, self.n - active_bit_level(l, self.n), -1):
            block_size = int(2 ** s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(self.B[j - branch_size, s])
                    self.B[j, s - 1] = self.B[j, s]
