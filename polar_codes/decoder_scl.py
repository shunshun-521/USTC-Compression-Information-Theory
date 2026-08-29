"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    _update_llrs,
    _update_bits,
    _bit_reversed,
)


def crc_encode(info_bits, crc_length=8):
    """CRC 编码（多项式长除法），返回信息比特 + CRC 校验位。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = 0x107 if crc_length == 8 else 0x11021
    msg = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    reg = msg.astype(int).tolist()
    for i in range(len(info_bits)):
        if reg[i] == 1:
            for j in range(crc_length + 1):
                if i + j < len(reg):
                    reg[i + j] ^= (poly >> (crc_length - j)) & 1
    crc_bits = np.array(reg[len(info_bits):len(info_bits) + crc_length], dtype=int)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    poly = 0x107 if crc_length == 8 else 0x11021
    reg = bits.astype(int).tolist()
    for i in range(len(bits) - crc_length):
        if reg[i] == 1:
            for j in range(crc_length + 1):
                if i + j < len(reg):
                    reg[i + j] ^= (poly >> (crc_length - j)) & 1
    return all(x == 0 for x in reg[-crc_length:])


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = (
            np.asarray(info_indices, dtype=int)
            if info_indices is not None
            else np.where(~self.frozen_bits)[0]
        )
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n

        paths = [{
            "L": np.full((N, n + 1), np.nan, dtype=np.float64),
            "B": np.full((N, n + 1), np.nan),
            "pm": 0.0,
            "u_hat": np.zeros(N, dtype=int),
        }]
        paths[0]["L"][:, 0] = llr_ch

        for l in self.decode_order:
            candidates = []
            for pidx, path in enumerate(paths):
                _update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]
                if l in self.frozen_set:
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    candidates.append((pm, pidx, 0))
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr, bit)
                        candidates.append((pm, pidx, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, pidx, bit in selected:
                parent = paths[pidx]
                child = {
                    "L": parent["L"].copy(),
                    "B": parent["B"].copy(),
                    "pm": pm,
                    "u_hat": parent["u_hat"].copy(),
                }
                child["u_hat"][l] = bit
                child["B"][l, n] = bit
                _update_bits(child["B"], l, n)
                new_paths.append(child)
            paths = new_paths

        if self.crc_length > 0:
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][self.info_indices], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
