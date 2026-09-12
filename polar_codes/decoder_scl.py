"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _sc_decode_recursive, f_operation, g_operation


def _crc_poly(crc_length):
    if crc_length == 8:
        return 0x07
    if crc_length == 16:
        return 0x8005
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 是否通过 CRC 校验"""
    bits = np.asarray(bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg == 0


def _path_metric(pm, llr, bit):
    hard = 0 if llr >= 0 else 1
    if bit != hard:
        pm += abs(llr)
    return pm


class SCLDecoder:
    """SCL 译码器：L=1 时与 SC 等价"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        if self.list_size == 1:
            u_hat, _ = _sc_decode_recursive(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        paths = [{"pm": 0.0, "llr": llr_ch.copy(), "u": np.zeros(self.N, dtype=np.int8)}]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_bit = self._leaf_llr(path["llr"], phi, path["u"][:phi])
                if self.frozen_bits[phi]:
                    new = {
                        "pm": _path_metric(path["pm"], llr_bit, 0),
                        "llr": path["llr"].copy(),
                        "u": path["u"].copy(),
                    }
                    new["u"][phi] = 0
                    candidates.append(new)
                else:
                    for bit in (0, 1):
                        new = {
                            "pm": _path_metric(path["pm"], llr_bit, bit),
                            "llr": path["llr"].copy(),
                            "u": path["u"].copy(),
                        }
                        new["u"][phi] = bit
                        candidates.append(new)

            candidates.sort(key=lambda p: p["pm"])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u"], self.crc_length)]
            best = min(valid, key=lambda p: p["pm"]) if valid else min(paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])
        return best["u"], best["pm"]

    def _leaf_llr(self, llr, phi, u_prefix):
        """计算第 phi 个比特的 LLR（考虑已译前缀比特）"""
        return self._leaf_llr_rec(llr, phi, self.n, u_prefix)

    def _leaf_llr_rec(self, llr, phi, stage, u_prefix=None):
        if stage == 0:
            return llr[phi]
        half = 1 << (stage - 1)
        if phi < half:
            left = f_operation(llr[:half], llr[half:])
            return self._leaf_llr_rec(left, phi, stage - 1, u_prefix)
        u_left = np.zeros(half, dtype=np.int8)
        if u_prefix is not None:
            u_left = u_prefix[:half].astype(np.int8)
        right = g_operation(llr[:half], llr[half:], u_left)
        return self._leaf_llr_rec(right, phi - half, stage - 1, u_prefix[half:phi] if u_prefix is not None else None)
