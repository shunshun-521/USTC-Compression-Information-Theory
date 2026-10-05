import numpy as np

from polar_scd_utils import (
    active_bit_level,
    active_llr_level,
    bit_reversed,
    hard_decision,
    lower_llr,
    upper_llr,
)


class PolarSCD:
    """SCD core (method bodies match polarcodes reference)."""

    def __init__(self, N, llr):
        self.myPC = _PC(N, llr)
        self.L = np.full((self.myPC.N, self.myPC.n + 1), np.nan, dtype=np.float64)
        self.B = np.full((self.myPC.N, self.myPC.n + 1), np.nan)
        self.L[:, 0] = self.myPC.likelihoods

    def decode(self, frozen_indices):
        self.myPC.frozen = np.asarray(frozen_indices, dtype=np.int64)
        for l in [bit_reversed(i, self.myPC.n) for i in range(self.myPC.N)]:
            self.update_llrs(l)
            if l in self.myPC.frozen:
                self.B[l, self.myPC.n] = 0
            else:
                self.B[l, self.myPC.n] = hard_decision(self.L[l, self.myPC.n])
            self.update_bits(l)
        return self.B[:, self.myPC.n].astype(np.int8)

    def update_llrs(self, l):
        for s in range(self.myPC.n - active_llr_level(l, self.myPC.n), self.myPC.n):
            block_size = int(2 ** (s + 1))
            branch_size = int(block_size / 2)
            for j in range(l, self.myPC.N, block_size):
                if j % block_size < branch_size:
                    top_llr = self.L[j, s]
                    btm_llr = self.L[j + branch_size, s]
                    self.L[j, s + 1] = upper_llr(top_llr, btm_llr)
                else:
                    btm_llr = self.L[j, s]
                    top_llr = self.L[j - branch_size, s]
                    top_bit = self.B[j - branch_size, s + 1]
                    self.L[j, s + 1] = lower_llr(btm_llr, top_llr, top_bit)

    def update_bits(self, l):
        if l < self.myPC.N / 2:
            return
        for s in range(self.myPC.n, self.myPC.n - active_bit_level(l, self.myPC.n), -1):
            block_size = int(2**s)
            branch_size = int(block_size / 2)
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    self.B[j - branch_size, s - 1] = int(self.B[j, s]) ^ int(
                        self.B[j - branch_size, s]
                    )
                    self.B[j, s - 1] = self.B[j, s]


class _PC:
    def __init__(self, N, llr):
        self.N = N
        self.n = int(np.log2(N))
        self.likelihoods = llr
        self.frozen = None
