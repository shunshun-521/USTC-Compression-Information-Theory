"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import copy
import numpy as np
from decoder_sc import (
    _compute_left_alpha,
    _compute_right_alpha,
    _compute_encoding_step,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    mask = (1 << crc_length) - 1
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        for _ in range(crc_length):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    rem = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=int
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC（bits 含信息位 + CRC）"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    payload = bits[:-crc_length]
    expected = crc_encode(payload, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected)


class _SCPath:
    """单条 SC 路径状态（与 decoder_sc.sc_decode 一致）"""

    def __init__(self, llr_ch, n, N, info_mask):
        self.n = n
        self.N = N
        self.info_mask = info_mask
        self.pm = 0.0
        self.intermediate_llr = [llr_ch.copy()]
        length = N // 2
        while length > 0:
            self.intermediate_llr.append(np.zeros(length, dtype=np.float64))
            length //= 2
        self.intermediate_bits = [np.zeros(N, dtype=int) for _ in range(n + 1)]
        self.current_state = np.zeros(n, dtype=int)
        self.previous_state = np.ones(n, dtype=int)

    def copy(self):
        return copy.deepcopy(self)

    def _update_llrs(self, position):
        bits = np.unpackbits(
            np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        self.current_state = bits[-self.n :].astype(int)
        for i in range(1, self.n + 1):
            if self.current_state[i - 1] == self.previous_state[i - 1]:
                continue
            llr = self.intermediate_llr[i - 1]
            if self.current_state[i - 1] == 0:
                self.intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - 2 ** (self.n - i)
                left_bits = self.intermediate_bits[i][start:end]
                self.intermediate_llr[i] = _compute_right_alpha(llr, left_bits)

    def _current_llr(self):
        return self.intermediate_llr[-1][0]

    def _apply_decision(self, position, bit):
        self.intermediate_bits[-1][position] = bit
        for i in range(self.n - 1, -1, -1):
            self.intermediate_bits[i] = _compute_encoding_step(
                i, self.n, self.intermediate_bits[i + 1], self.intermediate_bits[i]
            )
        self.previous_state = self.current_state.copy()

    @property
    def u_hat(self):
        return self.intermediate_bits[-1]


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时深拷贝）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.info_mask = ~self.frozen_bits
        self.L_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.info_mask)[0]

    @staticmethod
    def _pm_update(pm, llr, u):
        hard = 0 if llr >= 0 else 1
        return pm if u == hard else pm + abs(llr)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [_SCPath(llr_ch, self.n, self.N, self.info_mask)]

        for position in range(self.N):
            new_paths = []
            for path in paths:
                path._update_llrs(position)
                llr = path._current_llr()

                if not self.info_mask[position]:
                    child = path.copy()
                    child.pm = self._pm_update(child.pm, llr, 0)
                    child._apply_decision(position, 0)
                    new_paths.append(child)
                else:
                    for bit in (0, 1):
                        child = path.copy()
                        child.pm = self._pm_update(child.pm, llr, bit)
                        child._apply_decision(position, bit)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.L_size]

        crc_ok = []
        for p in paths:
            if self.crc_length > 0:
                bits = p.u_hat[self.info_positions]
                if crc_check(bits, self.crc_length):
                    crc_ok.append(p)
            else:
                crc_ok.append(p)

        pool = crc_ok if crc_ok else paths
        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.astype(int), best.pm
