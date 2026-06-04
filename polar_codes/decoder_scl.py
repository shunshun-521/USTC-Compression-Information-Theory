"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import _update_llrs, _update_bits, f_operation


def _bits_to_bytes(bits):
    """比特序列打包为字节（MSB 先行，末尾补零对齐）。"""
    bits = np.asarray(bits, dtype=int)
    pad = (-len(bits)) % 8
    if pad:
        bits = np.concatenate([bits, np.zeros(pad, dtype=int)])
    out = bytearray()
    for i in range(0, len(bits), 8):
        byte = 0
        for j in range(8):
            byte = (byte << 1) | int(bits[i + j])
        out.append(byte)
    return bytes(out)


def _crc_fun(crc_length):
    import crcmod
    if crc_length == 8:
        return crcmod.predefined.mkCrcFun("crc-8")
    return crcmod.predefined.mkCrcFun("crc-16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后（CRC-8: 0x07 / CRC-16: 0x8005）。"""
    info_bits = np.asarray(info_bits, dtype=int)
    crc_val = _crc_fun(crc_length)(_bits_to_bytes(info_bits))
    crc_bits = np.array(
        [(crc_val >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    return _crc_fun(crc_length)(_bits_to_bytes(bits)) == 0


def _pm_update(pm, llr, u):
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器（每条路径独立 L/B 数组）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.L_size = list_size
        self.crc_length = crc_length
        rev = bit_reversal_permutation(N)
        self.decode_order = [int(rev[i]) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.zeros((self.N, self.n + 1), dtype=np.float64)
        B = np.zeros((self.N, self.n + 1), dtype=int)
        L[:, 0] = llr_ch
        return {"pm": 0.0, "L": L, "B": B, "u_hat": np.zeros(self.N, dtype=int)}

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                L, B = path["L"], path["B"]
                _update_llrs(L, B, l, self.n, self.N)
                llr0 = L[l, self.n]

                if self.frozen_bits[l]:
                    path["pm"] = _pm_update(path["pm"], llr0, 0)
                    path["u_hat"][l] = 0
                    B[l, self.n] = 0
                    _update_bits(B, l, self.n, self.N)
                    new_paths.append(path)
                else:
                    for u_cand in (0, 1):
                        pcopy = {
                            "pm": _pm_update(path["pm"], llr0, u_cand),
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "u_hat": path["u_hat"].copy(),
                        }
                        pcopy["u_hat"][l] = u_cand
                        pcopy["B"][l, self.n] = u_cand
                        _update_bits(pcopy["B"], l, self.n, self.N)
                        new_paths.append(pcopy)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[: self.L_size]

        return self._select_best(paths)

    def _select_best(self, paths):
        if self.crc_length > 0:
            info_positions = np.where(~self.frozen_bits)[0]
            valid = [
                p for p in paths
                if crc_check(p["u_hat"][info_positions], self.crc_length)
            ]
            if valid:
                paths = valid
        best = min(paths, key=lambda p: p["pm"])
        return best["u_hat"].astype(int), best["pm"]
