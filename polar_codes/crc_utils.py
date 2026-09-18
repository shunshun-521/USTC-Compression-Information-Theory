"""CRC-8 / CRC-16 按比特反馈实现"""


def _crc_remainder(info_bits, crc_length):
    if crc_length == 8:
        poly = 0x07
        width = 8
    elif crc_length == 16:
        poly = 0x8005
        width = 16
    else:
        raise ValueError("crc_length must be 8 or 16")

    reg = 0
    top = 1 << (width - 1)
    mask = (1 << width) - 1
    for bit in info_bits:
        reg ^= int(bit) << (width - 1)
        for _ in range(8):
            if reg & top:
                reg = ((reg << 1) ^ poly) & mask
            else:
                reg = (reg << 1) & mask
    return reg


def crc_encode(info_bits, crc_length=8):
    info_bits = list(map(int, info_bits))
    rem = _crc_remainder(info_bits, crc_length)
    crc_bits = [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    import numpy as np
    return np.array(info_bits + crc_bits, dtype=int)


def crc_check(bits, crc_length=8):
    import numpy as np
    bits = np.asarray(bits, dtype=int)
    if crc_length == 0:
        return True
    rem = _crc_remainder(bits[:-crc_length], crc_length)
    expected = [(rem >> (crc_length - 1 - i)) & 1 for i in range(crc_length)]
    return np.array_equal(bits[-crc_length:], expected)
