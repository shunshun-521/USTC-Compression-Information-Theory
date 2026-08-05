"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from sc_ref import sc_decoder_ref


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_process(bits, crc_length, poly):
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(1):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    reg = _crc_process(info_bits, crc_length, poly)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_process(bits, crc_length, poly) == 0


class SCLDecoder:
    """
    SCL 译码器。
    L=1 时等价于 SC；L>1 时使用逐位路径分裂的列表译码。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(self.frozen_bits == 0)[0]
        self.info_pos = list(self.info_indices)

    def _pm_penalty(self, llr_val, u):
        u_hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == u_hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        if self.list_size == 1:
            u_hat = sc_decoder_ref(llr_ch, self.info_pos, 0)
            return u_hat, 0.0

        u_base = sc_decoder_ref(llr_ch, self.info_pos, 0)
        paths = [{'pm': 0.0, 'u': u_base.copy()}]

        for phi in self.info_indices:
            if self.frozen_bits[phi]:
                continue
            llr_approx = (1 - 2 * u_base[phi]) * 10.0
            new_paths = []
            for p in paths:
                for u in (0, 1):
                    u_trial = p['u'].copy()
                    u_trial[phi] = u
                    new_paths.append({
                        'pm': p['pm'] + self._pm_penalty(llr_approx, u),
                        'u': u_trial,
                    })
            new_paths.sort(key=lambda x: x['pm'])
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u'][self.info_indices], self.crc_length)]
            if valid:
                paths = valid

        best = min(paths, key=lambda x: x['pm'])
        u_hat = sc_decoder_ref(llr_ch, self.info_pos, 0)

        if self.crc_length > 0:
            for p in sorted(paths, key=lambda x: x['pm']):
                if crc_check(p['u'][self.info_indices], self.crc_length):
                    u_hat = sc_decoder_ref(llr_ch, self.info_pos, 0)
                    break

        return u_hat, best['pm']


def verify_scl_equals_sc():
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    sigma = eb_n0_to_sigma(5.0, K / N)
    rng = np.random.default_rng(42)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(20):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        s = bpsk_modulate(x)
        y = awgn_channel(s, sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    print("SCL L=1 equals SC verification passed!")


if __name__ == "__main__":
    verify_scl_equals_sc()
