"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np
from decoder_sc import _SCState, _bit_reversed

# ==================== CRC 工具 ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def _crc_mod2_divide(dividend, divisor):
    dividend = np.asarray(dividend, dtype=np.int8)
    divisor = np.asarray(divisor, dtype=np.int8)
    r = dividend.copy()
    for i in range(len(dividend) - len(divisor) + 1):
        if r[i] == 1:
            r[i : i + len(divisor)] ^= divisor
    return r[-(len(divisor) - 1) :]


def crc_encode(info_bits, crc_length=8):
    """
    计算 CRC 校验位并附加到信息比特后。
    """
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    divisor = np.array(
        [int(b) for b in format(poly, f"0{crc_length + 1}b")],
        dtype=np.int8,
    )
    payload = np.concatenate([info_bits, np.zeros(crc_length, dtype=np.int8)])
    rem = _crc_mod2_divide(payload, divisor)
    return np.concatenate([info_bits, rem])


def crc_check(bits, crc_length=8):
    """
    检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。
    """
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    divisor = np.array(
        [int(b) for b in format(poly, f"0{crc_length + 1}b")],
        dtype=np.int8,
    )
    rem = _crc_mod2_divide(bits, divisor)
    return np.all(rem == 0)


# ==================== SCL 译码器 ====================


class _Path:
    __slots__ = ("pm", "sc", "u_hat")

    def __init__(self, sc_state, u_hat=None):
        self.pm = 0.0
        self.sc = sc_state
        self.u_hat = (
            np.zeros(sc_state.N, dtype=np.int32)
            if u_hat is None
            else u_hat.copy()
        )

    def clone(self):
        new_sc = _SCState(self.sc.N, self.sc.frozen_bits)
        new_sc.L = self.sc.L.copy()
        new_sc.B = self.sc.B.copy()
        return _Path(new_sc, self.u_hat)


class SCLDecoder:
    """
    SCL 译码器（Lazy Copy：分裂时复制 LLR/比特树状态）。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def _pm_penalty(self, llr, bit):
        hard = 0 if llr >= 0 else 1
        return 0.0 if bit == hard else abs(llr)

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        base = _SCState(self.N, self.frozen_bits)
        base.set_channel(llr_ch)
        paths = [_Path(base)]

        for i in range(self.N):
            l = _bit_reversed(i, self.n)
            new_paths = []

            for path in paths:
                path.sc._update_llrs(l)
                llr = path.sc.L[l, self.n]

                if self.frozen_bits[l]:
                    path.pm += self._pm_penalty(llr, 0)
                    path.sc.B[l, self.n] = 0
                    path.u_hat[l] = 0
                    path.sc._update_bits(l)
                    new_paths.append(path)
                else:
                    for bit in (0, 1):
                        child = path.clone()
                        child.pm += self._pm_penalty(llr, bit)
                        child.sc.B[l, self.n] = bit
                        child.u_hat[l] = bit
                        child.sc._update_bits(l)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p.pm)
            paths = new_paths[: self.list_size]

        crc_valid = []
        for path in paths:
            if self.crc_length > 0:
                info_bits = path.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    crc_valid.append(path)

        if self.crc_length > 0 and crc_valid:
            best = min(crc_valid, key=lambda p: p.pm)
        else:
            best = paths[0]

        return best.u_hat.astype(int), float(best.pm)
