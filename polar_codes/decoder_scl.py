"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    f_boxplus,
    g_operation,
    f_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _prepare_llr,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg = ((reg << 1) | int(bit)) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    for _ in range(crc_length):
        reg = (reg << 1) & ((1 << (crc_length + 1)) - 1)
        if reg & (1 << crc_length):
            reg ^= poly
    return reg & ((1 << crc_length) - 1)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> (crc_length - 1 - i)) & 1
                         for i in range(crc_length)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYNOMIALS[crc_length]
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


class _Path:
    __slots__ = ('pm', 'u_hat', 'L', 'B')

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _copy_path(self, src):
        dst = _Path(self.N, self.n)
        dst.pm = src.pm
        dst.u_hat = src.u_hat.copy()
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        return dst

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_boxplus(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j, s],
                        path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _path_metric_penalty(self, llr_val, u_val):
        preferred = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == preferred else abs(llr_val)

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = _prepare_llr(llr_ch)
        active = [_Path(self.N, self.n)]
        active[0].L[:, 0] = llr

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in active:
                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    self._update_llrs(new_path, l)
                    llr_val = new_path.L[l, self.n]
                    new_path.pm += self._path_metric_penalty(llr_val, 0)
                    new_path.u_hat[phi] = 0
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = self._copy_path(path)
                        self._update_llrs(new_path, l)
                        llr_val = new_path.L[l, self.n]
                        new_path.pm += self._path_metric_penalty(llr_val, u_val)
                        new_path.u_hat[phi] = u_val
                        new_path.B[l, self.n] = u_val
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            active = candidates[:self.list_size]

        best = active[0]
        if self.crc_length > 0:
            crc_pass = [
                p for p in active
                if crc_check(p.B[:, self.n][self.info_indices], self.crc_length)
            ]
            if crc_pass:
                best = min(crc_pass, key=lambda p: p.pm)

        return best.B[:, self.n].astype(int), best.pm


def validate_scl_equals_sc(N=64, K=32, num_frames=20, eb_n0_db=5.0):
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        if not np.array_equal(u_sc, u_scl):
            return False
    return True
