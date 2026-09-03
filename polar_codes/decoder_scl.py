"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import (
    f_operation, g_operation, bit_reversed,
    active_llr_level, active_bit_level,
    _update_llrs, _update_bits,
)


_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, width):
    reg = 0
    for b in bits:
        reg ^= int(b) << (width - 1)
        for _ in range(8 if width <= 8 else 1):
            if width > 8:
                break
        if width <= 8:
            if reg & (1 << (width - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
            else:
                reg = (reg << 1) & ((1 << width) - 1)
    return reg

def _crc_process_byte(reg, byte, poly, width):
    reg ^= byte << (width - 8) if width > 8 else byte
    for _ in range(8):
        if reg & (1 << (width - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
        else:
            reg = (reg << 1) & ((1 << width) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
  """
  计算 CRC 校验位并附加到信息比特后。
  CRC-8 (0x07) 或 CRC-16 (0x8005)
  """
  info_bits = np.asarray(info_bits, dtype=np.int8)
  if crc_length == 8:
    poly, width = _CRC8_POLY, 8
  elif crc_length == 16:
    poly, width = _CRC16_POLY, 16
  else:
    raise ValueError("crc_length must be 8 or 16")

  reg = 0
  for bit in info_bits:
    reg ^= int(bit) << (width - 1)
    for _ in range(1):
      if reg & (1 << (width - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
      else:
        reg = (reg << 1) & ((1 << width) - 1)

  crc_bits = np.array([(reg >> (width - 1 - i)) & 1 for i in range(width)], dtype=np.int8)
  return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
  """检验 bits 末尾 CRC 是否正确"""
  bits = np.asarray(bits, dtype=np.int8)
  if crc_length == 8:
    poly, width = _CRC8_POLY, 8
  elif crc_length == 16:
    poly, width = _CRC16_POLY, 16
  else:
    raise ValueError("crc_length must be 8 or 16")

  reg = 0
  for bit in bits:
    reg ^= int(bit) << (width - 1)
    for _ in range(1):
      if reg & (1 << (width - 1)):
        reg = ((reg << 1) ^ poly) & ((1 << width) - 1)
      else:
        reg = (reg << 1) & ((1 << width) - 1)
  return reg == 0


class _Path:
    __slots__ = ('L', 'B', 'pm', 'u_hat')

    def __init__(self, N, n, llr_ch):
        self.L = np.full((N, n + 1), np.nan, dtype=np.float64)
        self.B = np.zeros((N, n + 1), dtype=np.int8)
        self.L[:, 0] = llr_ch
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=np.int8)

    def copy(self):
        p = object.__new__(_Path)
        p.L = self.L.copy()
        p.B = self.B.copy()
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        return p


class SCLDecoder:
    """SCL 译码器（Permuted SCD + 路径列表）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [bit_reversed(i, self.n) for i in range(N)]

    def _path_metric_update(self, pm, llr_val, u_bit):
        hard = 0 if llr_val >= 0 else 1
        if u_bit != hard:
            pm += abs(llr_val)
        return pm

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_Path(self.N, self.n, llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path.L, path.B, l, self.n, self.N)
                llr_l = path.L[l, self.n]

                if l in self.frozen_set:
                    p = path.copy()
                    p.pm = self._path_metric_update(p.pm, llr_l, 0)
                    p.B[l, self.n] = 0
                    p.u_hat[l] = 0
                    _update_bits(p.B, l, self.n, self.N)
                    new_paths.append(p)
                else:
                    for u_bit in (0, 1):
                        p = path.copy()
                        p.pm = self._path_metric_update(p.pm, llr_l, u_bit)
                        p.B[l, self.n] = u_bit
                        p.u_hat[l] = u_bit
                        _update_bits(p.B, l, self.n, self.N)
                        new_paths.append(p)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[:self.list_size]

        if self.crc_length > 0:
          info_positions = [i for i in self.decode_order if i not in self.frozen_set]
          valid = []
          for p in paths:
            payload = p.u_hat[info_positions]
            if crc_check(payload, self.crc_length):
              valid.append(p)
          if valid:
            paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm


if __name__ == "__main__":
    from encoder import polar_encode
    from construction import ga_construction
    from decoder_sc import sc_decode

    N, K = 64, 32
    info, _, _ = ga_construction(N, K, 2.5)
    frozen = np.ones(N, dtype=bool)
    frozen[info] = False

    rng = np.random.default_rng(1)
    errs_l1 = 0
    for _ in range(50):
        u = np.zeros(N, dtype=np.int8)
        u[info] = rng.integers(0, 2, K)
        x = polar_encode(u)
        llr = np.where(x == 0, 50.0, -50.0)
        u_sc = sc_decode(llr, frozen)
        u_scl, _ = SCLDecoder(N, frozen, list_size=1).decode(llr)
        if not np.array_equal(u_sc, u_scl):
            errs_l1 += 1
    print(f"L=1 vs SC mismatches: {errs_l1}/50")
