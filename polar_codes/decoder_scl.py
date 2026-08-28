"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _llr_at_phi, _prepare_channel_llr


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_poly(crc_length):
    if crc_length == 8:
        return CRC8_POLY
    if crc_length == 16:
        return CRC16_POLY
    raise ValueError("crc_length must be 8 or 16")


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _crc_poly(crc_length)
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        reg ^= int(bit) << (crc_length - 1)
        if reg & (1 << (crc_length - 1)):
            reg = ((reg << 1) ^ poly) & mask
        else:
            reg = (reg << 1) & mask
    crc_bits = np.array(
        [(reg >> i) & 1 for i in range(crc_length - 1, -1, -1)], dtype=np.int8
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    bits = np.asarray(bits, dtype=np.int8)
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _pm_update(pm, llr, u):
    """路径度量更新：不一致分支加 |LLR|。"""
    u_hard = 0 if llr >= 0 else 1
    if u != u_hard:
        pm += abs(llr)
    return pm


class Path:
    """单条译码路径。"""

    __slots__ = ("pm", "u_hat", "llr_br")

    def __init__(self, llr_br):
        self.pm = 0.0
        self.u_hat = np.zeros(len(llr_br), dtype=int)
        self.llr_br = llr_br


class SCLDecoder:
    """SCL 译码器。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """
        主译码函数。
        返回：u_hat, pm
        """
        llr_br = _prepare_channel_llr(llr_ch)
        paths = [Path(llr_br)]

        for phi in range(self.N):
            candidates = []
            for path in paths:
                llr_leaf = _llr_at_phi(path.llr_br, path.u_hat, phi)

                if self.frozen_bits[phi]:
                    child = Path(path.llr_br)
                    child.pm = _pm_update(path.pm, llr_leaf, 0)
                    child.u_hat = path.u_hat.copy()
                    child.u_hat[phi] = 0
                    candidates.append(child)
                else:
                    for bit in (0, 1):
                        child = Path(path.llr_br)
                        child.pm = _pm_update(path.pm, llr_leaf, bit)
                        child.u_hat = path.u_hat.copy()
                        child.u_hat[phi] = bit
                        candidates.append(child)

            candidates.sort(key=lambda p: p.pm)
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            valid = []
            for p in paths:
                info_bits = p.u_hat[self.info_indices]
                if crc_check(info_bits, self.crc_length):
                    valid.append(p)
            if valid:
                paths = valid

        best = min(paths, key=lambda p: p.pm)
        return best.u_hat.copy(), best.pm
