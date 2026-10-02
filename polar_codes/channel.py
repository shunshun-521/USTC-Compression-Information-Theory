"""
AWGN 信道 + BPSK 调制/解调
调制：0 -> +1, 1 -> -1
"""
import numpy as np


def bpsk_modulate(x):
    """将二进制码字 x (0/1) 映射为 BPSK 符号 (+1/-1)"""
    return 1 - 2 * np.asarray(x, dtype=np.float64)


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
    """
    return 2.0 * np.asarray(y, dtype=np.float64) / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    SNR = Eb/N0 * 2R（线性）
    sigma = 1 / sqrt(SNR) = 1 / sqrt(2R * 10^{Eb/N0/10})
    """
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def channel_llr_to_decoder(llr_ch):
    """将自然顺序信道 LLR 重排为译码器因子图顺序（比特倒序）。"""
    llr_ch = np.asarray(llr_ch, dtype=np.float64)
    n = len(llr_ch)
    idx = bit_reversal_permutation(n)
    return llr_ch[idx]


def decoder_llr_from_channel(llr_dec):
    """译码器顺序 LLR -> 自然顺序（channel_llr_to_decoder 的逆）。"""
    llr_dec = np.asarray(llr_dec, dtype=np.float64)
    n = len(llr_dec)
    idx = bit_reversal_permutation(n)
    out = np.empty(n, dtype=np.float64)
    out[idx] = llr_dec
    return out


def bit_reversal_permutation(N):
    """返回长度 N 的比特倒序置换索引数组"""
    n = int(np.log2(N))
    return np.array([int(f"{i:0{n}b}"[::-1], 2) for i in range(N)], dtype=int)
