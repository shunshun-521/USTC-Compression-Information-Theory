"""
极化码 SCL（串行抵消列表）译码器
from decoder_sc import (
    _update_llrs,
    _update_bits,
    _bit_reversed,
    _active_llr_level,
    _active_bit_level,
    _hard_decision,
    _path_metric_update,
)


CRC8_POLY = 0x07
CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后"""
    info_bits = np.asarray(info_bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in info_bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits 末尾 CRC 是否正确"""
    bits = np.asarray(bits, dtype=int)
    poly = CRC8_POLY if crc_length == 8 else CRC16_POLY
    mask = (1 << crc_length) - 1
    reg = 0
    for bit in bits:
        msb = (reg >> (crc_length - 1)) & 1
        reg = ((reg << 1) | int(bit)) & mask
        if msb ^ int(bit):
            reg ^= poly
    return reg == 0


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=int)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.decode_order = [_bit_reversed(i, self.n) for i in range(N)]

    def _new_path(self, llr_ch):
        L = np.full((self.N, self.n + 1), np.nan, dtype=np.float64)
        B = np.full((self.N, self.n + 1), np.nan)
        L[:, 0] = llr_ch
        return {'L': L, 'B': B, 'pm': 0.0, 'u_hat': np.zeros(self.N, dtype=int)}

    def _copy_path(self, path):
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
        }

    def decode(self, llr_ch):
        """主译码函数"""
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        paths = [self._new_path(llr_ch)]

        for l in self.decode_order:
            new_paths = []
            for path in paths:
                _update_llrs(path['L'], path['B'], l, self.n)
                leaf_llr = path['L'][l, self.n]

                if l in self.frozen_set:
                    child = self._copy_path(path)
                    child['pm'] = _path_metric_update(child['pm'], leaf_llr, 0)
                    child['u_hat'][l] = 0
                    child['B'][l, self.n] = 0
                    _update_bits(child['B'], l, self.n, self.N)
                    new_paths.append(child)
                else:
                    for u in (0, 1):
                        child = self._copy_path(path)
                        child['pm'] = _path_metric_update(child['pm'], leaf_llr, u)
                        child['u_hat'][l] = u
                        child['B'][l, self.n] = u
                        _update_bits(child['B'], l, self.n, self.N)
                        new_paths.append(child)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[: self.list_size]

        if self.crc_length > 0:
            valid = [p for p in paths if crc_check(p['u_hat'], self.crc_length)]
            best = min(valid, key=lambda p: p['pm']) if valid else min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].copy(), best['pm']
