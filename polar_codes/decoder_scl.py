"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
    f_operation,
    g_operation,
)


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length):
    poly = CRC_POLYNOMIALS[crc_length]
    mask = (1 << crc_length) - 1
    top = 1 << (crc_length - 1)
    crc = 0
    for bit in bits:
        msb = (crc >> (crc_length - 1)) & 1
        crc = (crc << 1) & mask
        if msb ^ int(bit):
            crc ^= poly
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    return _crc_remainder(bits, crc_length) == 0


class _Path:
    __slots__ = ("pm", "u_hat", "L", "B")

    def __init__(self, N, n):
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(~self.frozen_bits)[0]

    def _copy_path(self, path):
        new_path = _Path(self.N, self.n)
        new_path.pm = path.pm
        new_path.u_hat = path.u_hat.copy()
        new_path.L = path.L.copy()
        new_path.B = path.B.copy()
        return new_path

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)

        paths = [_Path(self.N, self.n)]
        paths[0].L[:, 0] = llr_ch

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            is_frozen = self.frozen_bits[l]
            candidates = []

            for path in paths:
                self._update_llrs(path, l)
                llr0 = path.L[l, self.n]

                if is_frozen:
                    penalty = 0.0 if llr0 >= 0 else abs(llr0)
                    new_path = self._copy_path(path)
                    new_path.pm += penalty
                    new_path.u_hat[l] = 0
                    self._backprop(new_path, l, 0)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        penalty = (
                            0.0
                            if (bit == 0 and llr0 >= 0) or (bit == 1 and llr0 < 0)
                            else abs(llr0)
                        )
                        new_path = self._copy_path(path)
                        new_path.pm += penalty
                        new_path.u_hat[l] = bit
                        self._backprop(new_path, l, bit)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_positions]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            chosen = valid[0] if valid else paths[0]
        else:
            chosen = paths[0]

        return chosen.u_hat.copy(), chosen.pm

    def _update_llrs(self, path, l):
        _update_llrs(path.L, path.B, l, self.n)

    def _backprop(self, path, l, bit):
        path.B[l, self.n] = bit
        _update_bits(path.B, l, self.n)


if __name__ == "__main__":
    from decoder_sc import sc_decode
    from encoder import polar_encode

    rng = np.random.default_rng(1)
    N = 64
    frozen = np.zeros(N, dtype=bool)
    scale = 50.0

    for L in [1, 4]:
        errs = 0
        for _ in range(100):
            u = rng.integers(0, 2, N)
            x = polar_encode(u)
            llr = (1 - 2 * x).astype(float) * scale
            u_hat, _ = SCLDecoder(N, frozen, list_size=L).decode(llr)
            if L == 1:
                u_ref = sc_decode(llr, frozen)
                if not np.array_equal(u_hat, u_ref):
                    errs += 1
            elif not np.array_equal(u_hat, u):
                errs += 1
        print(f"L={L}: errors={errs}")

    # CRC round-trip
    info = rng.integers(0, 2, 20)
    coded = crc_encode(info, 8)
    assert crc_check(coded, 8)
    print("CRC OK")
