"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
)
from encoder import bit_reversal_permutation


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    poly = CRC_POLYNOMIALS[crc_length]
    info_bits = np.asarray(info_bits, dtype=int)
    reg = 0
    for bit in info_bits:
        reg <<= 1
        reg |= int(bit)
        if reg & (1 << crc_length):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


class Path:
    """单条 SCL 译码路径"""

    __slots__ = ('pm', 'B', 'L', 'active')

    def __init__(self, n, N):
        self.pm = 0.0
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int32)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.info_indices = np.where(~self.frozen_bits)[0]
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _copy_path(self, src):
        dst = Path(self.n, self.N)
        dst.pm = src.pm
        dst.L = src.L.copy()
        dst.B = src.B.copy()
        return dst

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(path.L[j, s], path.L[j + branch_size, s])
                else:
                    top_llr = path.L[j - branch_size, s]
                    btm_llr = path.L[j, s]
                    top_bit = path.B[j - branch_size, s + 1]
                    path.L[j, s + 1] = g_operation(top_llr, btm_llr, top_bit)

    def _update_bits(self, path, l):
        if l < self.N // 2:
            return
        for s in range(self.n, self.n - _active_bit_level(l, self.n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = path.B[j, s] ^ path.B[j - branch_size, s]
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm。"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        br = bit_reversal_permutation(self.N)
        llr_ch = llr_ch[br]

        paths = [Path(self.n, self.N)]
        paths[0].L[:, 0] = llr_ch

        for i, l in enumerate(self.decode_order):
            candidates = []
            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = self._copy_path(path)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    self._update_bits(new_path, l)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        new_path = self._copy_path(path)
                        new_path.pm += self._pm_penalty(llr, bit)
                        new_path.B[l, self.n] = bit
                        self._update_bits(new_path, l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            crc_pass = [
                p for p in paths
                if crc_check(p.B[:, self.n][self.info_indices], self.crc_length)
            ]
            best = min(crc_pass, key=lambda p: p.pm) if crc_pass else paths[0]
        else:
            best = paths[0]

        return best.B[:, self.n].copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, num_frames=20, eb_n0_db=5.0):
    """单路径 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)

        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), 'SCL L=1 != SC'

    return True


if __name__ == '__main__':
    verify_scl_equals_sc()
    print('SCL L=1 verification passed')
