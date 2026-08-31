"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    return 1.0 - 2.0 * np.asarray(x, dtype=np.float64)


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    noise = rng.normal(0.0, sigma, size=np.shape(s))
    return s + noise


def compute_llr(y, sigma):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = ln P(y|x=0) / P(y|x=1) = 2*y / sigma^2
    返回形状与 y 相同的 LLR 数组
    """
    return 2.0 * np.asarray(y, dtype=np.float64) / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    SNR = Eb/N0 * 2R（线性）
    sigma = 1 / sqrt(SNR) = 1 / sqrt(2R * 10^{Eb/N0/10})
    """
    snr_linear = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def build_frozen_mask(N, info_indices):
    """
    构造 PSCD 译码器使用的冻结位掩码。
    PSCD 在树索引 l 处检查 frozen_bits[l]，与 GA 返回的 info_indices 直接对应。
    """
    frozen = np.ones(N, dtype=bool)
    frozen[np.asarray(info_indices, dtype=int)] = False
    return frozen


def reorder_channel_llr(llr_ch):
    """
    对信道 LLR 做比特倒序置换，与编码端 B_N 保持一致。
    译码器输入顺序与 u 向量索引一致。
    """
    N = len(llr_ch)
    br = np.array([int(f"{i:0{int(np.log2(N))}b}"[::-1], 2) for i in range(N)])
    return np.asarray(llr_ch, dtype=np.float64)[br]
