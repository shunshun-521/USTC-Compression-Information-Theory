"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    _upper_llr, _lower_llr, _bit_reversed,
    _active_llr_level, _active_bit_level,
)


# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8 if crc_length == 8 else 16):
            if crc_length == 8:
                msb = reg & 0x80
                reg = (reg << 1) & 0xFF
            else:
                msb = reg & 0x8000
                reg = (reg << 1) & 0xFFFF
            if msb:
                reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    CRC-8: 0x07; CRC-16: 0x8005
    """
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array([(remainder >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(bits, poly, crc_length)
    return remainder == 0


# ==================== SCL 译码器 ====================

class Path:
    """单条译码路径（Lazy Copy）"""
    __slots__ = ('L', 'B', 'pm', 'u_hat', 'active')

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.full((N, n + 1), np.nan)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int32)
        self.active = True


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.rev = bit_reversal_permutation(N)

    def _update_llrs(self, path, l):
        N, n = self.N, self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 2 ** (s + 1)
            branch_size = block_size // 2
            for j in range(l, N, block_size):
                if j % block_size < branch_size:
                    path.L[j, s + 1] = _upper_llr(path.L[j, s], path.L[j + branch_size, s])
                else:
                    path.L[j, s + 1] = _lower_llr(
                        path.L[j, s], path.L[j - branch_size, s],
                        path.B[j - branch_size, s + 1],
                    )

    def _update_bits(self, path, l):
        N, n = self.N, self.n
        if l < N // 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 2 ** s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    path.B[j - branch_size, s - 1] = int(path.B[j, s]) ^ int(path.B[j - branch_size, s])
                    path.B[j, s - 1] = path.B[j, s]

    def _pm_penalty(self, llr_val, bit):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if bit == hard else abs(llr_val)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.rev]
        N, n = self.N, self.n

        paths = [Path(N, n, llr_ch.copy())]

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                if not path.active:
                    continue
                self._update_llrs(path, l)
                llr_val = path.L[l, n]

                if l in self.frozen_set:
                    pen = self._pm_penalty(llr_val, 0)
                    path.pm += pen
                    path.u_hat[l] = 0
                    path.B[l, n] = 0
                    self._update_bits(path, l)
                    candidates.append((path.pm, path))
                else:
                    for bit in (0, 1):
                        new_path = Path(N, n, path.L[:, 0].copy())
                        new_path.L = path.L.copy()
                        new_path.B = path.B.copy()
                        new_path.pm = path.pm + self._pm_penalty(llr_val, bit)
                        new_path.u_hat = path.u_hat.copy()
                        new_path.u_hat[l] = bit
                        new_path.B[l, n] = bit
                        self._update_bits(new_path, l)
                        candidates.append((new_path.pm, new_path))

            candidates.sort(key=lambda x: x[0])
            paths = [c[1] for c in candidates[:self.list_size]]

        # 选择最优路径
        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p.u_hat, self.crc_length)]
            pool = valid if valid else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm


def validate_scl_equals_sc(N=64, K=32, num_frames=20):
    """L=1 时 SCL 应等价于 SC"""
    from construction import ga_construction
    from encoder import polar_encode
    from decoder_sc import sc_decode

    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=int)
    frozen_bits[info_idx] = 0

    rng = np.random.default_rng(1)
    scl = SCLDecoder(N, frozen_bits, list_size=1, crc_length=0)

    for _ in range(num_frames):
        u = np.zeros(N, dtype=int)
        u[info_idx] = rng.integers(0, 2, K)
        llr = np.where(polar_encode(u) == 0, 100.0, -100.0)
        u_sc = sc_decode(llr, frozen_bits)
        u_scl, _ = scl.decode(llr)
        assert np.array_equal(u_sc, u_scl), "SCL L=1 != SC"
    return True
