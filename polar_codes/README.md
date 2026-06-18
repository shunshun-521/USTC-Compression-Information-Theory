# 极化码编译码仿真

纯 Python（NumPy/SciPy）实现的极化码仿真框架，包含：

- GA 构造 (`construction.py`)
- 蝶形编码器 (`encoder.py`)
- AWGN 信道 (`channel.py`)
- SC / SCL / CA-SCL / BP 译码器
- 蒙特卡洛仿真与结果导出

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python3 verify.py          # 单元测试
python3 run_exp1.py        # SC 仿真
python3 run_exp2.py        # SCL / CA-SCL 仿真
python3 run_exp3.py        # BP 仿真
```

设置 `POLAR_QUICK=1` 可缩短仿真时间（减小码长与帧数）。

## 说明

- SC 译码采用置换 SC（Vangala 风格），与编码器输出比特倒序约定一致。
- 编码器验证：`N=4, u=[1,0,1,1] -> x=[1,0,1,1]`（与 `G_N=B_N F^{⊗n}` 矩阵乘法一致）。
