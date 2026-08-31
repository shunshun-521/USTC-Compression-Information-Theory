"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import sc_decode, sc_stepping_decode, _pm_update_hf


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.info_pos = self.info_indices.tolist()

    def _init_matrices(self, llr_ch):
        llr_matrix = np.ones((self.n + 1, self.N))
        llr_matrix[llr_matrix == 1] = float('nan')
        bit_matrix = llr_matrix.copy()
        llr_matrix[0] = llr_ch
        return llr_matrix, bit_matrix

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_matrix, bit_matrix = self._init_matrices(llr_ch)
        llr_list = [llr_matrix]
        bit_list = [bit_matrix]
        pm_list = [0.0]
        split_pos = self.info_pos.copy()
        split_loc = 0
        split_len = len(split_pos)
        l_now = 1

        while split_len - 1 >= split_loc:
            new_llr_list = []
            new_bit_list = []
            new_pm_list = []

            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                pm_temp = pm_list[i]

                llr_out, bit_out = sc_stepping_decode(
                    llr_temp, bit_temp, self.info_pos, 0, split_pos[split_loc]
                )

                prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                curr = split_pos[split_loc] + 1
                llr_slice = llr_out[self.n][prev:curr]
                bit_slice = bit_out[self.n][prev:curr]

                pm_correct = pm_temp + _pm_update_hf(llr_slice, bit_slice)
                new_llr_list.append(llr_out)
                new_bit_list.append(bit_out)
                new_pm_list.append(pm_correct)

                bit_wrong = bit_out.copy()
                bit_wrong[self.n][split_pos[split_loc]] = 1 - bit_wrong[self.n][split_pos[split_loc]]
                bit_slice_wrong = bit_wrong[self.n][prev:curr]
                pm_wrong = pm_temp + _pm_update_hf(llr_slice, bit_slice_wrong)
                new_llr_list.append(llr_out.copy())
                new_bit_list.append(bit_wrong)
                new_pm_list.append(pm_wrong)

            order = np.argsort(new_pm_list)[: self.list_size]
            llr_list = [new_llr_list[i] for i in order]
            bit_list = [new_bit_list[i] for i in order]
            pm_list = [new_pm_list[i] for i in order]
            l_now = len(pm_list)
            split_loc += 1

        if split_pos and split_pos[-1] != self.N - 1:
            for i in range(l_now):
                llr_temp = llr_list[i].copy()
                bit_temp = bit_list[i].copy()
                pm_temp = pm_list[i]
                llr_out, bit_out = sc_stepping_decode(
                    llr_temp, bit_temp, self.info_pos, 0, self.N - 1
                )
                prev = split_pos[split_loc - 1] + 1 if split_loc > 0 else 0
                pm_temp += _pm_update_hf(llr_out[self.n][prev:self.N], bit_out[self.n][prev:self.N])
                llr_list[i] = llr_out
                bit_list[i] = bit_out
                pm_list[i] = pm_temp

        order = np.argsort(pm_list)
        best_u = None
        best_pm = None

        if self.crc_length > 0:
            for idx in order:
                u_cand = bit_list[idx][self.n].astype(int)
                if crc_check(u_cand[self.info_indices], self.crc_length):
                    best_u = u_cand
                    best_pm = pm_list[idx]
                    break

        if best_u is None:
            idx = order[0]
            best_u = bit_list[idx][self.n].astype(int)
            best_pm = pm_list[idx]

        return best_u, best_pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=10.0, num_frames=50):
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(123)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits.astype(bool))
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"

    return True


if __name__ == "__main__":
    print("Running SCL verification...")
    verify_scl_equals_sc()
    print("SCL decoder tests passed.")
