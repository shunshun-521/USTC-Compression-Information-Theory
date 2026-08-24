"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _update_bits,
    _update_llrs,
)


CRC_POLYS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        fb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if fb ^ int(bit):
            reg ^= poly & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC"""
    bits = np.asarray(bits, dtype=int)
    if len(bits) < crc_length:
        return False
    encoded = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], encoded[-crc_length:])


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    @staticmethod
    def _pm_penalty(llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N = self.N
        n = self.n
        L = self.list_size

        paths = []
        L_arr = np.zeros((N, n + 1), dtype=np.float64)
        B_arr = np.zeros((N, n + 1), dtype=int)
        L_arr[:, 0] = llr_ch
        paths.append({"L": L_arr.copy(), "B": B_arr.copy(), "pm": 0.0, "u_hat": np.zeros(N, dtype=int)})

        for i in range(N):
            l = _bit_reversed(i, n)
            candidates = []

            for path in paths:
                _update_llrs(path["L"], path["B"], l, n, N)
                llr = path["L"][l, n]

                if self.frozen_bits[i]:
                    pm = path["pm"] + self._pm_penalty(llr, 0)
                    new_path = {
                        "L": path["L"].copy(),
                        "B": path["B"].copy(),
                        "pm": pm,
                        "u_hat": path["u_hat"].copy(),
                    }
                    new_path["B"][l, n] = 0
                    new_path["u_hat"][i] = 0
                    _update_bits(new_path["B"], l, n, N)
                    candidates.append(new_path)
                else:
                    for bit in (0, 1):
                        pm = path["pm"] + self._pm_penalty(llr, bit)
                        new_path = {
                            "L": path["L"].copy(),
                            "B": path["B"].copy(),
                            "pm": pm,
                            "u_hat": path["u_hat"].copy(),
                        }
                        new_path["B"][l, n] = bit
                        new_path["u_hat"][i] = bit
                        _update_bits(new_path["B"], l, n, N)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[:L]

        if self.crc_length > 0:
            valid = []
            for path in paths:
                info_bits = path["u_hat"][self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(path)
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
