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


def reorder_channel_llr(llr_ch, N):
    """
    将信道 LLR 按比特倒序重排，与编码器 B_N 置换一致。
    编码后码字经 B_N 置换，接收端 LLR 需做相同重排后送入译码器。
    """
    br = np.zeros(N, dtype=int)
    n = int(np.log2(N))
    for i in range(N):
        r = 0
        v = i
        for _ in range(n):
            r = (r << 1) | (v & 1)
            v >>= 1
        br[i] = r
    return llr_ch[br]


def hard_decision_llr(llr_ch):
    """LLR 硬判决为比特（0 if LLR >= 0 else 1）"""
    return (llr_ch < 0).astype(int)
