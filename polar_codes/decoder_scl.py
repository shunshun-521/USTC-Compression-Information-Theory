"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from encoder import bit_reversal_permutation
from decoder_sc import (
    SCDecoder,
    _compute_left_alpha,
    _compute_right_alpha,
    _compute_encoding_step,
)

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_remainder(bits, poly, crc_length):
    reg = 0
    for b in bits:
        reg ^= int(b) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
        else:
            reg = (reg << 1) & ((1 << crc_length) - 1)
    return reg


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    remainder = _crc_remainder(info_bits, poly, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """CRC 校验。"""
    bits = np.asarray(bits, dtype=int)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    payload = bits[:-crc_length]
    return np.array_equal(bits, crc_encode(payload, crc_length))


class _SCLPath:
    """单条 SCL 路径。"""

    def __init__(self, n, N, received_llr):
        self.n = n
        self.N = N
        self.pm = 0.0
        self.u_hat = np.zeros(N, dtype=int)
        self.current_state = np.zeros(n, dtype=np.int8)
        self.previous_state = np.ones(n, dtype=np.int8)
        self.intermediate_llr = [received_llr.copy()]
        length = N // 2
        while length > 0:
            self.intermediate_llr.append(np.zeros(length, dtype=np.float64))
            length //= 2
        self.intermediate_bits = [np.zeros(N, dtype=np.int8) for _ in range(n + 1)]

    def copy(self):
        p = _SCLPath(self.n, self.N, self.intermediate_llr[0])
        p.pm = self.pm
        p.u_hat = self.u_hat.copy()
        p.current_state = self.current_state.copy()
        p.previous_state = self.previous_state.copy()
        p.intermediate_llr = [x.copy() for x in self.intermediate_llr]
        p.intermediate_bits = [x.copy() for x in self.intermediate_bits]
        return p

    def _position_state(self, position):
        bits = np.unpackbits(
            np.array([position], dtype=np.uint32).byteswap().view(np.uint8)
        )
        return bits[-self.n :]

    def propagate_llr(self, position):
        n = self.n
        current_state = self._position_state(position)
        for i in range(1, n + 1):
            llr = self.intermediate_llr[i - 1]
            if current_state[i - 1] == self.previous_state[i - 1]:
                continue
            if current_state[i - 1] == 0:
                self.intermediate_llr[i] = _compute_left_alpha(llr)
            else:
                end = position
                start = end - (1 << (n - i))
                left_bits = self.intermediate_bits[i][start:end]
                self.intermediate_llr[i] = _compute_right_alpha(llr, left_bits)
        self.current_state = current_state

    def apply_bit(self, position, decision):
        n = self.n
        self.u_hat[position] = decision
        self.intermediate_bits[-1][position] = decision
        for i in range(n - 1, -1, -1):
            self.intermediate_bits[i] = _compute_encoding_step(
                i, n, self.intermediate_bits[i + 1], self.intermediate_bits[i]
            )
        self.previous_state = self.current_state.copy()


class SCLDecoder:
    """SCL 译码器（Lazy Copy：路径分裂时复制）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.mask = (self.frozen_bits == 0).astype(int)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_positions = np.where(self.frozen_bits == 0)[0]
        self._rev = bit_reversal_permutation(N)

    @staticmethod
    def _llr_penalty(llr_val, u):
        hard = 0 if llr_val >= 0 else 1
        return 0.0 if u == hard else abs(llr_val)

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)[self._rev]
        n, N = self.n, self.N
        L = self.list_size

        paths = [_SCLPath(n, N, llr_ch)]
        for position in range(N):
            candidates = []
            for path in paths:
                path.propagate_llr(position)
                cur_llr = path.intermediate_llr[-1][0]
                if self.mask[position] == 0:
                    pm = path.pm + self._llr_penalty(cur_llr, 0)
                    p = path.copy()
                    p.pm = pm
                    p.apply_bit(position, 0)
                    candidates.append(p)
                else:
                    for u in (0, 1):
                        p = path.copy()
                        p.pm = path.pm + self._llr_penalty(cur_llr, u)
                        p.apply_bit(position, u)
                        candidates.append(p)
            candidates.sort(key=lambda p: p.pm)
            paths = candidates[:L]

        if self.crc_length > 0:
            ok = [
                p
                for p in paths
                if crc_check(p.u_hat[self.info_positions], self.crc_length)
            ]
            pool = ok if ok else paths
        else:
            pool = paths

        best = min(pool, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
