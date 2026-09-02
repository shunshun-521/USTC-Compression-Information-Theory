"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math

import numpy as np

from decoder_sc import _SCDCore, _bit_reversed, sc_decode


CRC_POLYNOMIALS = {
    8: 0x07,
    16: 0x8005,
}


def crc_encode(info_bits, crc_length=8):
  """计算 CRC 校验位并附加到信息比特后（自洽校验和）。"""
  info_bits = np.asarray(info_bits, dtype=np.int8)
  checksum = 0
  for idx, bit in enumerate(info_bits):
    checksum ^= int(bit) << (idx % crc_length)
  checksum &= (1 << crc_length) - 1
  crc_bits = np.array(
    [(checksum >> i) & 1 for i in range(crc_length)],
    dtype=np.int8,
  )
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
  if crc_length == 0:
    return True
  bits = np.asarray(bits, dtype=np.int8)
  expected = crc_encode(bits[:-crc_length], crc_length)[-crc_length:]
  return np.array_equal(bits[-crc_length:], expected)


class SCLDecoder:
    """SCL 译码器（基于 Permuted SCD）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]
        self.frozen_set = set(np.where(self.frozen_bits)[0])

    def decode(self, llr_ch):
        """主译码函数。"""
        if self.list_size == 1:
            u_hat = sc_decode(llr_ch, self.frozen_bits)
            return u_hat, 0.0

        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [{"pm": 0.0, "core": _SCDCore(self.N, llr_ch, self.frozen_bits), "u_hat": np.zeros(self.N, dtype=np.int8)}]

        for l in self.decode_order:
            active = []
            for path in paths:
                core = path["core"]
                core._update_llrs(l)
                llr = core.L[l, self.n]

                if l in self.frozen_set:
                    new_core = _SCDCore(self.N, llr_ch, self.frozen_bits)
                    new_core.L = core.L.copy()
                    new_core.B = core.B.copy()
                    new_core.B[l, self.n] = 0
                    new_core._update_bits(l)
                    active.append(
                        {
                            "pm": path["pm"] + (abs(llr) if llr < 0 else 0.0),
                            "core": new_core,
                            "u_hat": path["u_hat"].copy(),
                        }
                    )
                    active[-1]["u_hat"][l] = 0
                else:
                    hard = 0 if llr >= 0 else 1
                    for bit in (0, 1):
                        new_core = _SCDCore(self.N, llr_ch, self.frozen_bits)
                        new_core.L = core.L.copy()
                        new_core.B = core.B.copy()
                        new_core.B[l, self.n] = bit
                        new_core._update_bits(l)
                        active.append(
                            {
                                "pm": path["pm"] + (0.0 if bit == hard else abs(llr)),
                                "core": new_core,
                                "u_hat": path["u_hat"].copy(),
                            }
                        )
                        active[-1]["u_hat"][l] = bit

            active.sort(key=lambda p: p["pm"])
            paths = active[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p["u_hat"], self.crc_length)]
            best = min(valid if valid else paths, key=lambda p: p["pm"])
        else:
            best = min(paths, key=lambda p: p["pm"])

        return best["u_hat"].copy(), best["pm"]
