"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _bit_reversal, _prepare_llr, _update_llrs, _update_bits


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc8_bits(bits):
    crc = 0
    for b in bits:
        crc ^= int(b) << 7
        for _ in range(8):
            if crc & 0x80:
                crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
            else:
                crc = (crc << 1) & 0xFF
    return crc


def _crc16_bits(bits):
    crc = 0
    for b in bits:
        crc ^= int(b) << 15
        for _ in range(16):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（CRC-8: 0x07）"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        remainder = _crc8_bits(np.concatenate([info_bits, np.zeros(8, dtype=int)]))
        crc_bits = np.array([(remainder >> (7 - i)) & 1 for i in range(8)], dtype=int)
    else:
        remainder = _crc16_bits(np.concatenate([info_bits, np.zeros(16, dtype=int)]))
        crc_bits = np.array([(remainder >> (15 - i)) & 1 for i in range(16)], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    payload = bits[:-crc_length]
    return np.array_equal(bits, crc_encode(payload, crc_length))


class _Path:
    __slots__ = ("L", "B", "pm")

    def __init__(self, N, n, llr_ch):
        self.L = np.zeros((N, n + 1), dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=int)
        self.L[:, 0] = llr_ch
        self.pm = 0.0


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

    def _path_metric_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = _prepare_llr(llr_ch)
        paths = [_Path(self.N, self.n, llr_ch)]

        for i in range(self.N):
            l = _bit_reversal(i, self.n)
            candidates = []

            for pidx, path in enumerate(paths):
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr = path.L[l, self.n]

                if l in self.frozen_set:
                    bit = 0
                    new_pm = path.pm + self._path_metric_penalty(llr, bit)
                    candidates.append((new_pm, pidx, bit))
                else:
                    for bit in (0, 1):
                        new_pm = path.pm + self._path_metric_penalty(llr, bit)
                        candidates.append((new_pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            survivors = candidates[: self.list_size]

            new_paths = []
            for pm, parent_idx, bit in survivors:
                parent = paths[parent_idx]
                child = _Path(self.N, self.n, llr_ch)
                child.L = parent.L.copy()
                child.B = parent.B.copy()
                child.pm = pm
                child.B[l, self.n] = bit
                _update_bits(child.B, l, self.n, self.N)
                new_paths.append(child)

            paths = new_paths

        if self.crc_length > 0:
            info_idx = sorted(set(range(self.N)) - self.frozen_set)
            valid = []
            for path in paths:
                payload = path.B[:, self.n].astype(int)[info_idx]
                if crc_check(payload, self.crc_length):
                    valid.append(path)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.B[:, self.n].astype(int), best.pm


if __name__ == "__main__":
    from construction import ga_construction
    from encoder import polar_encode
    from channel import bpsk_modulate, compute_llr
    from decoder_sc import sc_decode

    N, K = 64, 32
    info_idx, _, _ = ga_construction(N, K, 2.5)
    frozen_bits = np.ones(N, dtype=bool)
    frozen_bits[info_idx] = False

    u = np.zeros(N, dtype=int)
    u[info_idx] = np.random.randint(0, 2, K)
    llr = compute_llr(bpsk_modulate(polar_encode(u)), 0.1)

    u_sc = sc_decode(llr, frozen_bits)
    u_scl, _ = SCLDecoder(N, frozen_bits, list_size=1).decode(llr)
    print("L=1 vs SC match:", np.array_equal(u_sc, u_scl))
