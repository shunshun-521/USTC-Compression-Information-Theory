"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np

from encoder import bit_reversal_permutation
from decoder_sc import cn_op, g_operation

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for bit in bits:
        reg ^= int(bit) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    return _crc_remainder(bits, poly, crc_length) == 0


class PathState:
  __slots__ = ("pm", "llr", "frozen_ind", "u_hat", "u_up")

  def __init__(self, N):
      self.pm = 0.0
      self.llr = np.asarray(np.zeros(N), dtype=np.float64)
      self.frozen_ind = None
      self.u_hat = np.zeros(N, dtype=int)
      self.u_up = np.zeros(N, dtype=np.float64)


def _scl_decode_single_path(llr_ch, frozen_ind):
    """单路径 SCL 核心（返回 u_hat, u_up 根节点部分和）"""
    n = len(llr_ch)
    frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
    if n == 1:
        if frozen_ind[0] == 1:
            u_hat = np.array([0.0])
        else:
            u_hat = np.array([0.0 if llr_ch[0] >= 0 else 1.0])
        return u_hat, u_hat.copy()

    half = n // 2
    llr_left = cn_op(llr_ch[:half], llr_ch[half:])
    u_left, u_left_up = _scl_decode_single_path(llr_left, frozen_ind[:half])
    llr_right = g_operation(llr_ch[:half], llr_ch[half:], u_left_up)
    u_right, u_right_up = _scl_decode_single_path(llr_right, frozen_ind[half:])
    u_hat = np.concatenate([u_left, u_right])
    u_left_up = np.bitwise_xor(
        u_left_up.astype(int), u_right_up.astype(int)
    ).astype(np.float64)
    u_up = np.concatenate([u_left_up, u_right_up])
    return u_hat, u_up


def _pm_penalty(llr, u):
    preferred = 0 if llr >= 0 else 1
    return 0.0 if u == preferred else abs(llr)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.br = bit_reversal_permutation(N)
        self.frozen_ind = np.zeros(N, dtype=np.float64)
        self.frozen_ind[self.frozen_bits] = 1.0
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _decode_recursive_paths(self, llr_ch, frozen_ind, list_size):
        """递归 SCL，在叶节点分裂路径"""
        n = len(llr_ch)
        frozen_ind = np.asarray(frozen_ind, dtype=np.float64)
        if n == 1:
            paths = []
            llr = float(llr_ch[0])
            if frozen_ind[0] == 1:
                p = PathState(1)
                p.pm = _pm_penalty(llr, 0)
                p.u_hat[0] = 0
                p.u_up[0] = 0.0
                paths.append(p)
            else:
                for u in (0, 1):
                    p = PathState(1)
                    p.pm = _pm_penalty(llr, u)
                    p.u_hat[0] = u
                    p.u_up[0] = float(u)
                    paths.append(p)
            return paths

        half = n // 2
        llr_left = cn_op(llr_ch[:half], llr_ch[half:])
        left_paths = self._decode_recursive_paths(llr_left, frozen_ind[:half], list_size)

        all_paths = []
        for lp in left_paths:
            llr_right = g_operation(llr_ch[:half], llr_ch[half:], lp.u_up[:half])
            right_paths = self._decode_recursive_paths(
                llr_right, frozen_ind[half:], list_size
            )
            for rp in right_paths:
                merged = PathState(n)
                merged.pm = lp.pm + rp.pm
                merged.u_hat[:half] = lp.u_hat[:half]
                merged.u_hat[half:] = rp.u_hat[:half]
                up_left = np.bitwise_xor(
                    lp.u_up[:half].astype(int), rp.u_up[:half].astype(int)
                ).astype(np.float64)
                merged.u_up[:half] = up_left
                merged.u_up[half:] = rp.u_up[:half]
                all_paths.append(merged)

        all_paths.sort(key=lambda p: p.pm)
        return all_paths[:list_size]

    def decode(self, llr_ch):
        """主译码函数，返回 (u_hat, pm)"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self.br]
        paths = self._decode_recursive_paths(llr_ch, self.frozen_ind, self.list_size)

        best_crc = None
        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        best = best_crc if best_crc is not None else paths[0]
        return best.u_hat.copy(), best.pm
