"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    x = np.asarray(x, dtype=int)
    return 1 - 2 * x


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        rng = np.random.default_rng()
    s = np.asarray(s, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=s.shape)
    return s + noise


def compute_llr(y, sigma):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = 2*y / sigma^2
    """
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * y / (sigma ** 2)


def align_llr_for_decoder(llr_ch, N):
    """
    编码器在码字上施加比特倒序置换时，将信道 LLR 映射到 SC/SCL 译码器索引顺序。
    """
    n = int(np.log2(N))
    rev = np.array([int(format(i, f"0{n}b")[::-1], 2) for i in range(N)], dtype=int)
    return np.asarray(llr_ch, dtype=np.float64)[rev]


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    """
    snr_linear = 2.0 * rate * (10 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)
