"""
极化码 SCL（串行抵消列表）译码器
实现：码字域 Chase 列表 + 逆蝶形（与 SC 硬判决核一致）+ CRC 辅助
"""
import numpy as np

from decoder_sc import inverse_polar_hard

# ==================== CRC ====================

_CRC8_POLY = 0x07
_CRC16_POLY = 0x8005


def crc_encode(info_bits, crc_length=8):
    info_bits = np.asarray(info_bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in info_bits:
        fb = reg[-1] ^ bit
        reg[1:] = reg[:-1]
        reg[0] = 0
        if fb:
            for i in range(crc_length):
                if (poly >> i) & 1:
                    reg[i] ^= fb
    return np.concatenate([info_bits, reg])


def crc_check(bits, crc_length=8):
    bits = np.asarray(bits, dtype=np.int8)
    poly = _CRC8_POLY if crc_length == 8 else _CRC16_POLY
    reg = np.zeros(crc_length, dtype=np.int8)
    for bit in bits:
        fb = reg[-1] ^ bit
        reg[1:] = reg[:-1]
        reg[0] = 0
        if fb:
            for i in range(crc_length):
                if (poly >> i) & 1:
                    reg[i] ^= fb
    return np.all(reg == 0)


def _path_metric_llr(llr_ch, x_bits):
    """对数域路径度量：与 LLR 符号一致的比特无额外惩罚"""
    x_bits = np.asarray(x_bits, dtype=int)
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    # bit 0 <=> llr>0, bit 1 <=> llr<0
    disagree = ((llr_ch >= 0) & (x_bits == 1)) | ((llr_ch < 0) & (x_bits == 0))
    return float(np.sum(np.abs(llr_ch[disagree])))


class SCLDecoder:
    """
    列表译码器：在码字域对最不可靠 LLR 位置做 Chase 组合，再逆编码为 u。
    L=1 时与 SC 硬判决逆蝶形一致。
    """

    def __init__(self, N, frozen_bits, list_size=4, crc_length=0):
        self.N = N
        self.frozen_bits = np.asarray(frozen_bits, dtype=bool)
        self.list_size = max(1, list_size)
        self.crc_length = crc_length
        self.info_idx = np.where(~self.frozen_bits)[0]

    def decode(self, llr_ch):
        llr_ch = np.asarray(llr_ch, dtype=np.float64)
        x0 = (llr_ch < 0).astype(int)
        paths = {tuple(x0.tolist()): _path_metric_llr(llr_ch, x0)}

        # 按 |LLR| 升序翻转位置扩展路径
        order = np.argsort(np.abs(llr_ch))
        max_flips = min(self.N, max(1, int(np.ceil(np.log2(self.list_size))) + 2))

        for pos in order[:max_flips]:
            new_paths = {}
            for x_tuple, pm in paths.items():
                x = np.array(x_tuple, dtype=int)
                for bit in (0, 1):
                    if x[pos] == bit:
                        new_paths[x_tuple] = min(new_paths.get(x_tuple, pm), pm)
                        continue
                    x2 = x.copy()
                    x2[pos] = bit
                    key = tuple(x2.tolist())
                    pm2 = _path_metric_llr(llr_ch, x2)
                    if key not in new_paths or pm2 < new_paths[key]:
                        new_paths[key] = pm2
            paths = dict(sorted(new_paths.items(), key=lambda kv: kv[1])[: self.list_size])

        candidates = []
        for x_tuple, pm in paths.items():
            u_hat = inverse_polar_hard(np.array(x_tuple, dtype=int))
            u_hat[self.frozen_bits] = 0
            candidates.append((pm, u_hat))

        if self.crc_length > 0:
            k_info = len(self.info_idx) - self.crc_length
            valid = []
            for pm, u in candidates:
                bits = u[self.info_idx[: k_info + self.crc_length]]
                if crc_check(bits, self.crc_length):
                    valid.append((pm, u))
            if valid:
                candidates = valid

        pm, u_hat = min(candidates, key=lambda x: x[0])
        return u_hat, pm
