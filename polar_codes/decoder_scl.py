"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _scd_update_bits,
    _scd_update_llrs,
    active_bit_level,
    active_llr_level,
)
from encoder import bit_reversed


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def _crc_remainder(bits, crc_length=8):
    poly = CRC_POLYNOMIALS[crc_length]
    msg = list(map(int, bits))
    for i in range(len(bits) - crc_length):
        if msg[i]:
            for j in range(crc_length + 1):
                if (poly >> j) & 1:
                    msg[i + j] ^= 1
    return np.array(msg[-crc_length:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    padded = np.concatenate([info_bits, np.zeros(crc_length, dtype=int)])
    remainder = _crc_remainder(padded, crc_length)
    return np.concatenate([info_bits, remainder])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    return np.all(_crc_remainder(bits, crc_length) == 0)


class SCLDecoder:
    """SCL 译码器（路径复制实现）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.frozen_set = set(np.where(self.frozen_bits > 0)[0])

    def _pm_update(self, pm, llr, bit):
        hard = 0 if llr >= 0 else 1
        if bit != hard:
            pm += abs(llr)
        return pm

    def decode(self, llr_ch):
        """主译码函数。返回：u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        n = self.n
        N = self.N
        L_size = self.list_size

        def new_path():
            return {
                "pm": 0.0,
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
            }

        paths = [new_path()]
        paths[0]["L"][:, 0] = llr_ch

        for phi in range(N):
            l = bit_reversed(phi, n)
            new_paths = []
            for path in paths:
                _scd_update_llrs(path["L"], path["B"], l, n)
                llr = path["L"][l, n]
                if l in self.frozen_set:
                    pm = self._pm_update(path["pm"], llr, 0)
                    new_p = {
                        "pm": pm,
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                    }
                    new_p["B"][l, n] = 0
                    _scd_update_bits(new_p["B"], l, n)
                    new_p["u_partial"] = path.get(
                        "u_partial", np.zeros(N, dtype=int)
                    ).copy()
                    new_paths.append(new_p)
                else:
                    for bit in (0, 1):
                        pm = self._pm_update(path["pm"], llr, bit)
                        new_p = {
                            "pm": pm,
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                        }
                        new_p["B"][l, n] = bit
                        _scd_update_bits(new_p["B"], l, n)
                        new_paths.append(new_p)

            new_paths.sort(key=lambda p: p["pm"])
            paths = new_paths[:L_size]

        best_crc = None
        best_any = min(paths, key=lambda p: p["pm"])
        if self.crc_length > 0:
            crc_paths = []
            for p in paths:
                u_hat = p["B"][:, n].astype(int)
                info_mask = self.frozen_bits == 0
                info_bits = u_hat[info_mask]
                if crc_check(info_bits, self.crc_length):
                    crc_paths.append(p)
            if crc_paths:
                best_crc = min(crc_paths, key=lambda p: p["pm"])

        chosen = best_crc if best_crc is not None else best_any
        u_hat = chosen["B"][:, n].astype(int)
        return u_hat, chosen["pm"]
