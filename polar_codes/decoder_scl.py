"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed_index,
    _map_channel_llrs,
    _prepare_frozen,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_update(reg, bit, crc_length, poly):
    reg ^= int(bit) << (crc_length - 1)
    mask = (1 << crc_length) - 1
    if reg & (1 << (crc_length - 1)):
        reg = ((reg << 1) ^ poly) & mask
    else:
        reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in info_bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    crc_bits = np.array([(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=int).ravel()
    poly = CRC_POLYNOMIALS[crc_length]
    reg = 0
    for bit in bits:
        reg = _crc_update(reg, bit, crc_length, poly)
    return reg == 0


class _Path:
    __slots__ = ('pm', 'L', 'B')

    def __init__(self, N, n, llr):
        self.pm = 0.0
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr.copy()


class SCLDecoder:
    """SCL 译码器（Lazy Copy）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_set = set(_prepare_frozen(frozen_bits))
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.array(sorted(set(range(N)) - self.frozen_set), dtype=int)
        self.decode_order = [_bit_reversed_index(i, self.n) for i in range(N)]

    def _clone_path(self, path):
        new_path = _Path(self.N, self.n, np.zeros(self.N))
        new_path.pm = path.pm
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        return new_path

    def _pm_penalty(self, llr, u_bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u_bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr = _map_channel_llrs(llr_ch, self.N)
        paths = [_Path(self.N, self.n, llr)]

        for l in self.decode_order:
            candidates = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, f_operation)
                llr_val = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._clone_path(path)
                    new_path.pm += self._pm_penalty(llr_val, 0)
                    new_path.B[l, self.n] = 0
                    _update_bits(new_path.B, l, self.n)
                    candidates.append(new_path)
                else:
                    for u_bit in (0, 1):
                        new_path = self._clone_path(path)
                        new_path.pm += self._pm_penalty(llr_val, u_bit)
                        new_path.B[l, self.n] = u_bit
                        _update_bits(new_path.B, l, self.n)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        decoded_paths = []
        for path in paths:
            u_hat = path.B[:, self.n].astype(int)
            decoded_paths.append((path.pm, u_hat))

        if self.crc_length > 0:
            valid = [(pm, u) for pm, u in decoded_paths if crc_check(u[self.info_indices], self.crc_length)]
            pm, u_hat = min(valid or decoded_paths, key=lambda x: x[0])
        else:
            pm, u_hat = min(decoded_paths, key=lambda x: x[0])

        return u_hat.copy(), pm


def run_scl_self_test():
    """SCL L=1 应等价于 SC"""
    from channel import awgn_channel, bpsk_modulate, compute_llr, eb_n0_to_sigma
    from construction import ga_construction
    from encoder import polar_encode
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(456)
    sigma = eb_n0_to_sigma(4.0, K / N)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(50):
        u_src = np.zeros(N, dtype=int)
        u_src[info_idx] = rng.integers(0, 2, size=K)
        x = polar_encode(u_src)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 与 SC 不一致"

    info = rng.integers(0, 2, size=K - 8)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8), "CRC 编解码失败"
    print("SCL self-test passed.")
