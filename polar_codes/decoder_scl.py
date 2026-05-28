"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    upper_llr,
    lower_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


# CRC-8: x^8 + x^2 + x + 1 (0x07)
_CRC8_POLY = 0x07
# CRC-16: CRC-16-IBM (0x8005)
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, reg_bits):
    reg = 0
    mask = (1 << reg_bits) - 1
    for b in bits:
        reg ^= (int(b) << (reg_bits - 1))
        if reg & (1 << (reg_bits - 1)):
            reg = ((reg << 1) & mask) ^ poly
        else:
            reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int).ravel()
    if crc_length == 8:
        poly, reg_bits = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    rem = _crc_remainder(info_bits, poly, reg_bits)
    crc_bits = np.array(
        [(rem >> (reg_bits - 1 - i)) & 1 for i in range(reg_bits)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC。"""
    bits = np.asarray(bits, dtype=int).ravel()
    if crc_length == 8:
        poly, reg_bits = _CRC8_POLY, 8
    elif crc_length == 16:
        poly, reg_bits = _CRC16_POLY, 16
    else:
        raise ValueError("crc_length must be 8 or 16")
    rem = _crc_remainder(bits, poly, reg_bits)
    return rem == 0


class _Path:
  __slots__ = ("L", "B", "pm", "active")

  def __init__(self, N, n):
    self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
    self.B = np.zeros((N, n + 1), dtype=int)
    self.pm = 0.0
    self.active = True


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制 LLR/比特数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _llr_penalty(self, llr, u):
        v = 0 if llr >= 0 else 1
        return 0.0 if u == v else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        paths = [_Path(N, n) for _ in range(self.L_size)]
        paths[0].L[:, 0] = llr_ch.copy()
        active_count = 1

        decode_order = [_bit_reversed_index(i, n) for i in range(N)]

        for l in decode_order:
            candidates = []

            for p in paths[:active_count]:
                if not p.active:
                    continue

                self._update_llrs_path(p, l)

                cur_llr = p.L[l, n]
                if np.isnan(cur_llr):
                    cur_llr = 0.0

                if l in self.frozen_set:
                    pen = self._llr_penalty(cur_llr, 0)
                    p.pm += pen
                    p.B[l, n] = 0
                    self._update_bits_path(p, l)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        if len(candidates) >= self.L_size * 2:
                            break
                        cp = self._clone_path(p)
                        cp.pm += self._llr_penalty(cur_llr, u)
                        cp.B[l, n] = u
                        self._update_bits_path(cp, l)
                        candidates.append(cp)

            candidates.sort(key=lambda x: x.pm)
            active_count = min(len(candidates), self.L_size)
            paths[:active_count] = candidates[:active_count]
            for i in range(active_count, self.L_size):
                paths[i].active = False

        best = min(paths[:active_count], key=lambda x: x.pm)

        if self.crc_length > 0:
            valid = []
            for p in paths[:active_count]:
                u = p.B[:, n].astype(int)
                info_bits = u[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda x: x.pm)

        u_hat = best.B[:, n].astype(int)
        return u_hat, best.pm

    def _clone_path(self, p):
        q = _Path(self.N, self.n)
        q.L = p.L.copy()
        q.B = p.B.copy()
        q.pm = p.pm
        q.active = True
        return q

    def _update_llrs_path(self, p, l):
        n = self.n
        for s in range(n - _active_llr_level(l, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l, self.N, block_size):
                if j % block_size < branch_size:
                    p.L[j, s + 1] = upper_llr(p.L[j, s], p.L[j + branch_size, s])
                else:
                    p.L[j, s + 1] = lower_llr(
                        p.L[j, s],
                        p.L[j - branch_size, s],
                        p.B[j - branch_size, s + 1],
                    )

    def _update_bits_path(self, p, l):
        if l < self.N // 2:
            return
        n = self.n
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    p.B[j - branch_size, s - 1] = int(p.B[j, s]) ^ int(
                        p.B[j - branch_size, s]
                    )
                    p.B[j, s - 1] = p.B[j, s]


if __name__ == "__main__":
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N = 32
    frozen = np.zeros(N, dtype=bool)
    frozen[:16] = True
    u = np.zeros(N, dtype=int)
    u[~frozen] = np.random.randint(0, 2, np.sum(~frozen))
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 1e-6)
    uh_sc = sc_decode(llr, frozen)
    uh_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
    print("SCL L=1 vs SC:", np.array_equal(uh_sc, uh_scl))
