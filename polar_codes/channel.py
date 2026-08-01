"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    return 1 - 2 * x


def awgn_channel(s, sigma, rng=None):
    """加高斯白噪声，返回接收信号 y = s + n，n ~ N(0, sigma^2)"""
    if rng is None:
        noise = np.random.normal(0, sigma, size=s.shape)
    else:
        noise = rng.normal(0, sigma, size=s.shape)
    return s + noise


def compute_llr(y, sigma):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = ln P(y|x=0) / P(y|x=1) = 2*y / sigma^2
    """
    return 2.0 * y / sigma ** 2


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    SNR = Eb/N0 * 2R（线性）
    sigma = 1 / sqrt(SNR) = 1 / sqrt(2R * 10^{Eb/N0/10})
    """
    snr_linear = 2.0 * rate * 10 ** (eb_n0_db / 10.0)
    return 1.0 / np.sqrt(snr_linear)


def bit_reversal_permutation(N):
    """返回比特倒序置换索引数组"""
    n = int(np.log2(N))
    rev = np.zeros(N, dtype=int)
    for i in range(N):
        rev[i] = int(f"{i:0{n}b}"[::-1], 2)
    return rev


def permute_llr_for_decoder(llr_ch, N):
    """
    将信道顺序的 LLR 映射为 SC/SCL 译码器内部顺序。
    编码器输出含比特倒序置换，译码前需对 LLR 做相同置换。
    """
    rev = bit_reversal_permutation(N)
    return llr_ch[rev]
