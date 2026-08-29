"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _prepare_llr,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)

CRC8_POLY = 0x07
CRC16_POLY = 0x8005

_CRC8_TABLE = None


def _get_crc8_table():
    global _CRC8_TABLE
    if _CRC8_TABLE is None:
        table = []
        for i in range(256):
            crc = i
            for _ in range(8):
                if crc & 0x80:
                    crc = ((crc << 1) ^ CRC8_POLY) & 0xFF
                else:
                    crc = (crc << 1) & 0xFF
            table.append(crc)
        _CRC8_TABLE = table
    return _CRC8_TABLE


def _bits_to_bytes(bits):
    """将比特序列按 MSB-first 打包为字节"""
    bits = np.asarray(bits, dtype=int)
    nbytes = (len(bits) + 7) // 8
    out = []
    for i in range(nbytes):
        byte = 0
        for j in range(8):
            idx = i * 8 + j
            byte = (byte << 1) | (int(bits[idx]) if idx < len(bits) else 0)
        out.append(byte)
    return out


def _crc8_value(bits):
    table = _get_crc8_table()
    crc = 0
    for byte in _bits_to_bytes(bits):
        crc = table[crc ^ byte]
    return crc


def _crc16_value(bits):
    crc = 0
    mask = 0xFFFF
    for byte in _bits_to_bytes(bits):
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ CRC16_POLY) & mask
            else:
                crc = (crc << 1) & mask
    return crc


def _crc_value(bits, crc_length):
    return _crc8_value(bits) if crc_length == 8 else _crc16_value(bits)


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=int)
    remainder = _crc_value(info_bits, crc_length)
    crc_bits = np.array(
        [(remainder >> (crc_length - 1 - i)) & 1 for i in range(crc_length)],
        dtype=int,
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 CRC 是否正确。"""
    bits = np.asarray(bits, dtype=int)
    info = bits[:-crc_length]
    expected_crc = crc_encode(info, crc_length)[-crc_length:]
    return np.array_equal(bits[-crc_length:], expected_crc)


class SCLDecoder:
    """SCL 译码器（含 Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length

    def _path_penalty(self, llr_val, u_val):
        """路径度量惩罚：与 LLR 符号不一致时加 |LLR|"""
        preferred = 0 if llr_val >= 0 else 1
        return 0.0 if u_val == preferred else abs(llr_val)

    def _update_llrs(self, L, B, l_idx):
        n = self.n
        N = self.N
        for s in range(n - _active_llr_level(l_idx, n), n):
            block_size = 1 << (s + 1)
            branch_size = block_size // 2
            for j in range(l_idx, N, block_size):
                if j % block_size < branch_size:
                    L[j, s + 1] = f_operation(L[j, s], L[j + branch_size, s])
                else:
                    L[j, s + 1] = g_operation(
                        L[j - branch_size, s],
                        L[j, s],
                        B[j - branch_size, s + 1],
                    )

    def _update_bits(self, B, l_idx):
        n = self.n
        N = self.N
        if l_idx < N // 2:
            return
        for s in range(n, n - _active_bit_level(l_idx, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l_idx, -1, -block_size):
                if j % block_size >= branch_size:
                    B[j - branch_size, s - 1] = (B[j, s] + B[j - branch_size, s]) % 2
                    B[j, s - 1] = B[j, s]

    def decode(self, llr_ch):
        """主译码函数，返回 u_hat, pm"""
        N = self.N
        n = self.n
        llr = _prepare_llr(llr_ch)

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int32),
            'PM': 0.0,
            'u_hat': np.zeros(N, dtype=int),
        }]
        paths[0]['L'][:, 0] = llr

        for phi in range(N):
            l_idx = _bit_reversed_index(phi, n)
            candidates = []

            for path in paths:
                self._update_llrs(path['L'], path['B'], l_idx)
                llr_val = path['L'][l_idx, n]

                if l_idx in self.frozen_set:
                    new_path = {
                        'L': path['L'].copy(),
                        'B': path['B'].copy(),
                        'PM': path['PM'] + self._path_penalty(llr_val, 0),
                        'u_hat': path['u_hat'].copy(),
                    }
                    new_path['u_hat'][l_idx] = 0
                    new_path['B'][l_idx, n] = 0
                    self._update_bits(new_path['B'], l_idx)
                    candidates.append(new_path)
                else:
                    for u_val in (0, 1):
                        new_path = {
                            'L': path['L'].copy(),
                            'B': path['B'].copy(),
                            'PM': path['PM'] + self._path_penalty(llr_val, u_val),
                            'u_hat': path['u_hat'].copy(),
                        }
                        new_path['u_hat'][l_idx] = u_val
                        new_path['B'][l_idx, n] = u_val
                        self._update_bits(new_path['B'], l_idx)
                        candidates.append(new_path)

            candidates.sort(key=lambda p: p['PM'])
            paths = candidates[: self.list_size]

        if self.crc_length > 0:
            info_mask = ~self.frozen_bits
            valid = [
                p for p in paths
                if crc_check(p['u_hat'][info_mask], self.crc_length)
            ]
            best = min(valid if valid else paths, key=lambda p: p['PM'])
        else:
            best = min(paths, key=lambda p: p['PM'])

        return best['u_hat'], best['PM']
