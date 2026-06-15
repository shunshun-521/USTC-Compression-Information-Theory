"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import _preprocess_channel_llr
from sc_core import scl_decoder

# ==================== CRC 工具 ====================


def _crc_poly(crc_length):
    if crc_length == 8:
        loc = [8, 2, 1, 0]
    elif crc_length == 16:
        loc = [16, 15, 2, 0]
    else:
        raise ValueError(f"Unsupported CRC length: {crc_length}")
    p = [0] * (crc_length + 1)
    for i in loc:
        p[i] = 1
    return p[::-1]


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = [int(b) for b in np.asarray(info_bits, dtype=int)]
    p = _crc_poly(crc_length)
    info = info_bits.copy()
    times = len(info)
    for _ in range(crc_length):
        info.append(0)
    for i in range(times):
        if info[i] == 1:
            for j in range(crc_length + 1):
                info[j + i] ^= p[j]
    check_code = info[-crc_length:]
    return np.array(info_bits + check_code, dtype=int)


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确"""
    bits = [int(b) for b in np.asarray(bits, dtype=int)]
    info_len = len(bits) - crc_length
    info = bits[:info_len]
    recoded = crc_encode(info, crc_length)
    return np.array_equal(recoded, np.array(bits, dtype=int))


# ==================== SCL 译码器 ====================


class SCLDecoder:
    """SCL 译码器"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = list_size
        self.crc_length = crc_length
        self.information_pos = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        y_llr = _preprocess_channel_llr(llr_ch)
        crc_fn = None
        if self.crc_length > 0:
            crc_len = self.crc_length

            def crc_fn(bits):
                return crc_check(bits, crc_len)

        return scl_decoder(
            y_llr,
            self.information_pos,
            0,
            self.list_size,
            crc_check_fn=crc_fn,
        )
