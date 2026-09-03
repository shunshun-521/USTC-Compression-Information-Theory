"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import math
import numpy as np
from decoder_sc import (
    _SCDState,
    _active_bit_level,
    _active_llr_level,
    _bit_reversed,
    _lower_llr,
    _upper_llr,
)
from encoder import bit_reversal_permutation


CRC_POLYS = {8: 0x07, 16: 0x8005}


def _crc_process(bits, crc_length, poly, reg=0):
    mask = (1 << crc_length) - 1
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb:
            reg ^= poly
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    reg = _crc_process(
        np.concatenate([info_bits, np.zeros(crc_length, dtype=int)]),
        crc_length,
        poly,
    )
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC_POLYS[crc_length]
    return _crc_process(bits, crc_length, poly) == 0


class _SCLPath:
    """SCL 路径（Lazy Copy）"""

    __slots__ = ("pm", "state", "u_hat")

    def __init__(self, N, n, llr_ch):
        self.pm = 0.0
        self.state = _SCDState(N, n, llr_ch)
        self.u_hat = np.zeros(N, dtype=int)

    def clone(self):
        new = _SCLPath.__new__(_SCLPath)
        new.pm = self.pm
        new.state = self.state
        new.u_hat = self.u_hat.copy()
        return new


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(math.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]
        self._br = bit_reversal_permutation(N)

    def _pm_penalty(self, llr, u):
        hard = 0 if llr >= 0 else 1
        return 0.0 if u == hard else abs(llr)

    def _copy_state(self, src_state):
        """深拷贝 SCD 状态（Lazy Copy 在分裂时调用）"""
        st = _SCDState(self.N, self.n, src_state.L[:, 0])
        st.L = src_state.L.copy()
        st.B = src_state.B.copy()
        return st

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self._br]
        paths = [_SCLPath(self.N, self.n, llr_ch)]

        for phi in range(self.N):
            l = _bit_reversed(phi, self.n)
            candidates = []

            for path in paths:
                path.state.update_llrs(l)
                llr = path.state.L[l, self.n]

                if self.frozen_bits[l]:
                    new_path = path.clone()
                    new_path.state = self._copy_state(path.state)
                    new_path.pm += self._pm_penalty(llr, 0)
                    new_path.state.B[l, self.n] = 0
                    new_path.u_hat[l] = 0
                    new_path.state.update_bits(l)
                    candidates.append(new_path)
                else:
                    for u in (0, 1):
                        new_path = path.clone()
                        new_path.state = self._copy_state(path.state)
                        new_path.pm += self._pm_penalty(llr, u)
                        new_path.state.B[l, self.n] = u
                        new_path.u_hat[l] = u
                        new_path.state.update_bits(l)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        best_crc = None
        best_all = paths[0]

        if self.crc_length > 0:
            for path in paths:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    if best_crc is None or path.pm < best_crc.pm:
                        best_crc = path

        chosen = best_crc if best_crc is not None else best_all
        return chosen.u_hat.copy(), chosen.pm
