# Polar Codes Simulation

极化码（Polar Codes）编译码仿真：GA 构造、SC/SCL/CA-SCL/BP 译码、蒙特卡洛 BLER 仿真。

## 目录

| 文件 | 说明 |
|------|------|
| `construction.py` | 高斯近似（GA）极化码构造 |
| `encoder.py` | 蝶形编码 + 比特倒序 |
| `decoder_sc.py` | SC 译码（递归 + 非递归） |
| `decoder_scl.py` | SCL / CA-SCL 译码 |
| `decoder_bp.py` | BP 译码（min-sum + 早停） |
| `channel.py` | BPSK-AWGN 信道 |
| `simulation.py` | 蒙特卡洛仿真 |
| `utils.py` | CSV/绘图/容量限 |
| `run_exp1.py` | 实验一：SC |
| `run_exp2.py` | 实验二：SCL / CA-SCL |
| `run_exp3.py` | 实验三：BP 对比 |
| `verify.py` | 单元测试 |

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python3 verify.py          # 单元测试（约 1 分钟）
python3 run_exp1.py        # SC 仿真（约 10–30 分钟）
python3 run_exp2.py        # SCL 仿真（约 30–90 分钟，N=512 较慢）
python3 run_exp3.py        # BP 仿真（约 30–90 分钟）
# 或一键运行全部：
./run_all.sh
```

## 约定

- LLR：正号表示倾向 bit 0，`LLR = 2y/σ²`
- BPSK：0→+1，1→−1
- `frozen_bits`：1=冻结，0=信息
- 编码器：`u=[1,0,1,1] → x=[1,0,1,1]`（含比特倒序）

## GA 构造参考值

- N=8, K=4, Eb/N0=2.5 dB：`info=[0,3,5,6]`, `frozen=[1,2,4,7]`
- N=256, K=128：info 前 20 个见 `verify.py` 输出
