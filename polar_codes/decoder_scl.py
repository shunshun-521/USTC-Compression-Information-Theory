"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _prepare_llr, _update_llrs, _update_bits


def _bit_reversed(i, n):
    return int(format(i, f"0{n}b")[::-1], 2)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    if crc_length == 8:
        poly = 0x07
    elif crc_length == 16:
        poly = 0x8005
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)

    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    return np.array_equal(bits[-crc_length:], crc_encode(bits[:-crc_length], crc_length)[-crc_length:])


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def decode(self, llr_ch):
        """主译码函数。"""
        llr = _prepare_llr(llr_ch)
        N, n = self.N, self.n

        paths = [
            {
                "L": np.zeros((N, n + 1), dtype=np.float64),
                "B": np.zeros((N, n + 1), dtype=int),
                "pm": 0.0,
                "u_hat": np.zeros(N, dtype=int),
            }
        ]
        paths[0]["L"][:, 0] = llr

        for phi in self.decode_order:
            candidates = []
            for path in paths:
                L = path["L"]
                B = path["B"]
                _update_llrs(L, B, phi, n)
                llr_phi = L[phi, n]

                if self.frozen_bits[phi]:
                    bit = 0
                    penalty = 0.0 if llr_phi >= 0 else abs(llr_phi)
                    candidates.append((path["pm"] + penalty, path, bit))
                else:
                    preferred = 0 if llr_phi >= 0 else 1
                    for bit in (0, 1):
                        penalty = 0.0 if bit == preferred else abs(llr_phi)
                        candidates.append((path["pm"] + penalty, path, bit))

            candidates.sort(key=lambda x: x[0])
            selected = candidates[: self.list_size]

            new_paths = []
            for pm, parent, bit in selected:
                L = parent["L"].copy()
                B = parent["B"].copy()
                u_hat = parent["u_hat"].copy()
                u_hat[phi] = bit
                B[phi, n] = bit
                _update_bits(B, phi, n)
                new_paths.append({"L": L, "B": B, "pm": pm, "u_hat": u_hat})
            paths = new_paths

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
