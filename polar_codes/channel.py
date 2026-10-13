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
    s = np.asarray(s, dtype=np.float64)
    noise = rng.normal(0.0, sigma, size=s.shape)
    return s + noise


def compute_llr(y, sigma):
    """
    计算 BPSK-AWGN 信道的信道 LLR。
    LLR(y) = ln P(y|x=0) / P(y|x=1) = 2*y / sigma^2
    """
    y = np.asarray(y, dtype=np.float64)
    return 2.0 * y / (sigma ** 2)


def eb_n0_to_sigma(eb_n0_db, rate):
    """
    将 Eb/N0 (dB) 转换为 AWGN 噪声标准差 sigma。
    SNR = Eb/N0 * 2R（线性）
    sigma = 1 / sqrt(SNR) = 1 / sqrt(2R * 10^{Eb/N0/10})
    """
    snr_linear = 2.0 * rate * (10.0 ** (eb_n0_db / 10.0))
    return 1.0 / np.sqrt(snr_linear)


def bit_reversal_indices(N):
    """长度 N 的比特倒序置换索引：out[i] = bit_reverse(i)。"""
    n = int(np.log2(N))
    idx = np.arange(N, dtype=np.int64)
    bits = (idx[:, None] >> np.arange(n)) & 1
    return (bits[:, ::-1] * (1 << np.arange(n))).sum(axis=1)


def map_llr_to_decoder(llr_ch, N):
    """
    将按自然顺序的信道 LLR 映射到 SC/SCL 译码器输入顺序（比特倒序）。
    接收端 x[br[i]] 对应 u[i]，故 llr_dec[i] = llr_ch[br[i]]。
    """
    br = bit_reversal_indices(N)
    return llr_ch[br]


def map_llr_to_channel(llr_dec, N):
    """译码器顺序 LLR -> 信道码字 x 的自然顺序。"""
    br = bit_reversal_indices(N)
    out = np.empty(N, dtype=np.float64)
    out[br] = llr_dec
    return out
