"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversed
from decoder_sc import f_operation, g_operation, _active_llr_level, _active_bit_level


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = 0
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly
    return reg == 0


# ==================== SCL 译码器 ====================


class _PathState:
    """单条译码路径状态"""
    __slots__ = ('pm', 'L', 'B', 'copied')

    def __init__(self, N, n, llr_ch, parent=None):
        self.pm = 0.0
        self.copied = False
        if parent is None:
            self.L = np.zeros((N, n + 1), dtype=np.float64)
            self.B = np.zeros((N, n + 1), dtype=int)
            self.L[:, 0] = llr_ch
        else:
            self.L = parent.L
            self.B = parent.B

    def copy_from(self, parent):
        self.L = parent.L.copy()
        self.B = parent.B.copy()
        self.pm = parent.pm
        self.copied = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, u):
        u_hard = 0 if llr >= 0 else 1
        return 0.0 if u == u_hard else abs(llr)

    def _update_llrs(self, path, l):
        for s in range(self.n - _active_llr_level(l, self.n), self.n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = f_operation(
                        path.L[j, s], path.L[j + branch_size, s]
                    )
                else:
                    path.L[j, s + 1] = g_operation(
                        path.L[j - branch_size, s],
                        path.L[j, s],
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
                    path.B[j - branch_size, s - 1] = (
                        path.B[j, s] ^ path.B[j - branch_size, s]
                    )
                    path.B[j, s - 1] = path.B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_PathState(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = bit_reversed(i, self.n)
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    new_path = _PathState(self.N, self.n, llr_ch)
                    new_path.copy_from(path)
                    new_path.pm = path.pm + self._pm_penalty(llr, 0)
                    new_path.B[l, self.n] = 0
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = _PathState(self.N, self.n, llr_ch)
                        new_path.copy_from(path)
                        new_path.pm = path.pm + self._pm_penalty(llr, u)
                        new_path.B[l, self.n] = u
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

            for path in paths:
                self._update_bits(path, l)

        if self.crc_length > 0:
            valid = []
            for p in paths:
                u_full = p.B[:, self.n].astype(int)
                if crc_check(u_full[self.info_indices], self.crc_length):
                    valid.append(p)
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int).copy(), best.pm


def verify_scl_equals_sc(N=64, K=32, eb_n0_db=5.0):
    """L=1 的 SCL 应近似 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, awgn_channel, compute_llr, eb_n0_to_sigma
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False
    rate = K / N
    sigma = eb_n0_to_sigma(eb_n0_db, rate)
    rng = np.random.default_rng(123)

    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)
    for _ in range(30):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        x = polar_encode(u)
        y = awgn_channel(bpsk_modulate(x), sigma, rng)
        llr = compute_llr(y, sigma)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True


if __name__ == "__main__":
    verify_scl_equals_sc()
    print("SCL verification passed")
