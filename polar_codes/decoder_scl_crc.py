"""CRC 辅助工具。"""
import numpy as np

CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def _crc_generator(crc_length):
    if crc_length == 8:
        value = CRC8_POLY
    elif crc_length == 16:
        value = CRC16_POLY
    else:
        raise ValueError("crc_length must be 8 or 16")
    bits = format(value, f"0{crc_length}b")
    return [1] + [int(b) for b in bits]


def _crc_mod(message, generator):
    msg = message.tolist()
    r = len(generator) - 1
    padded = msg + [0] * r
    for i in range(len(message)):
        if padded[i]:
            for j in range(len(generator)):
                padded[i + j] ^= generator[j]
    return np.array(padded[-r:], dtype=int)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    generator = _crc_generator(crc_length)
    crc_bits = _crc_mod(info_bits, generator)
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    generator = _crc_generator(crc_length)
    return np.all(_crc_mod(bits, generator) == 0)
