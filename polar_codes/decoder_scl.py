"""
极化码 SCL（串行抵消列表）译码器
支持 CRC 辅助（CA-SCL）
"""
import numpy as np

from decoder_sc import (
    f_operation,
    g_operation,
    _bit_reversed_index,
    _active_llr_level,
    _active_bit_level,
)


# CRC-8: 0x07, CRC-16: 0x8005
_CRC_POLYS = {8: 0x07, 16: 0x8005}


def crc_encode(info_bits, crc_length=8):
    """计算 CRC 校验位并附加到信息比特后。"""
    info_bits = np.asarray(info_bits, dtype=np.int32)
    poly = _CRC_POLYS[crc_length]
    reg = 0
    for bit in info_bits:
        reg ^= (int(bit) << (crc_length - 1))
        for _ in range(8):
            if reg & (1 << (crc_length - 1)):
                reg = ((reg << 1) ^ poly) & ((1 << crc_length) - 1)
            else:
                reg = (reg << 1) & ((1 << crc_length) - 1)
    crc_bits = np.array(
        [(reg >> (crc_length - 1 - i)) & 1 for i in range(crc_length)], dtype=np.int32
    )
    return np.concatenate([info_bits, crc_bits])


def crc_check(bits, crc_length=8):
    """检验 bits[-r:] 是否是 bits[:-r] 的正确 CRC。"""
    if crc_length == 0:
        return True
    expected = crc_encode(bits[:-crc_length], crc_length)
    return np.array_equal(bits[-crc_length:], expected[-crc_length:])


def _path_metric_update(pm, llr, u):
    """路径度量更新：与 LLR 符号不一致时加 |LLR| 惩罚。"""
    u_hard = 0 if llr >= 0 else 1
    penalty = 0.0 if u == u_hard else abs(llr)
    return pm + penalty


class SCLDecoder:
    """SCL 译码器（Lazy Copy 优化）。"""

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0, info_indices=None):
        self.N = N
        self.n = int(np.log2(N))
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.frozen_set = set(np.where(self.frozen_bits)[0])
        self.list_size = list_size
        self.crc_length = crc_length
        self.info_indices = info_indices

    def decode(self, llr_ch):
        """
        主译码函数。
        llr_ch 应为 prepare_channel_llr 处理后的 LLR。
        """
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        N, n = self.N, self.n
        L = list_size = self.list_size

        paths = [{
            'L': np.zeros((N, n + 1), dtype=np.float64),
            'B': np.zeros((N, n + 1), dtype=np.int32),
            'pm': 0.0,
            'u_hat': np.zeros(N, dtype=np.int32),
            'active': True,
        }]
        paths[0]['L'][:, 0] = llr_ch

        for phi in range(N):
            l = _bit_reversed_index(phi, n)
            new_paths = []

            for path in paths:
                if not path['active']:
                    continue

                L_arr, B_arr = path['L'], path['B']

                for s in range(n - _active_llr_level(l, n), n):
                    block_size = 1 << (s + 1)
                    branch_size = block_size // 2
                    for j in range(l, N, block_size):
                        if j % block_size < branch_size:
                            L_arr[j, s + 1] = f_operation(L_arr[j, s], L_arr[j + branch_size, s])
                        else:
                            L_arr[j, s + 1] = g_operation(
                                L_arr[j - branch_size, s], L_arr[j, s],
                                B_arr[j - branch_size, s + 1]
                            )

                cur_llr = L_arr[l, n]

                if l in self.frozen_set:
                    pm = _path_metric_update(path['pm'], cur_llr, 0)
                    np_path = self._lazy_copy(path)
                    np_path['pm'] = pm
                    np_path['u_hat'][l] = 0
                    np_path['B'][l, n] = 0
                    self._update_bits(np_path, l)
                    new_paths.append(np_path)
                else:
                    for u_cand in (0, 1):
                        pm = _path_metric_update(path['pm'], cur_llr, u_cand)
                        np_path = self._lazy_copy(path)
                        np_path['pm'] = pm
                        np_path['u_hat'][l] = u_cand
                        np_path['B'][l, n] = u_cand
                        self._update_bits(np_path, l)
                        new_paths.append(np_path)

            new_paths.sort(key=lambda p: p['pm'])
            paths = new_paths[:L]
            for p in paths:
                p['active'] = True

        if self.crc_length > 0:
            valid = []
            for p in paths:
                bits = (p['u_hat'][self.info_indices] if self.info_indices is not None
                        else p['u_hat'])
                if crc_check(bits, self.crc_length):
                    valid.append(p)
            if valid:
                best = min(valid, key=lambda p: p['pm'])
            else:
                best = min(paths, key=lambda p: p['pm'])
        else:
            best = min(paths, key=lambda p: p['pm'])

        return best['u_hat'].astype(int), best['pm']

    def _lazy_copy(self, path):
        """Lazy copy：复制引用，写时由 numpy 自动处理。"""
        return {
            'L': path['L'].copy(),
            'B': path['B'].copy(),
            'pm': path['pm'],
            'u_hat': path['u_hat'].copy(),
            'active': True,
        }

    def _update_bits(self, path, l):
        B_arr = path['B']
        n, N = self.n, self.N
        if l < N / 2:
            return
        for s in range(n, n - _active_bit_level(l, n), -1):
            block_size = 1 << s
            branch_size = block_size // 2
            for j in range(l, -1, -block_size):
                if j % block_size >= branch_size:
                    B_arr[j - branch_size, s - 1] = int(B_arr[j, s]) ^ int(B_arr[j - branch_size, s])
                    B_arr[j, s - 1] = B_arr[j, s]
