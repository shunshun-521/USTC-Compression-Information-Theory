# 极化码编译码仿真

纯 Python（NumPy/SciPy）实现的极化码仿真框架，不依赖第三方极化码库。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python3 validate.py          # 单元测试
python3 run_exp1.py          # 实验一：SC
python3 run_exp2.py          # 实验二：SCL / CA-SCL
python3 run_exp3.py          # 实验三：BP
```

## 快速仿真模式

```bash
POLAR_FAST_SIM=1 python3 run_exp1.py
POLAR_FAST_SIM=1 python3 run_exp2.py
POLAR_FAST_SIM=1 python3 run_exp3.py
```

完整仿真（默认 `max_frames=100000`, `min_errors=100`）耗时较长，建议在性能较好的机器上运行。

## 目录说明

| 文件 | 功能 |
|------|------|
| `construction.py` | GA 高斯近似构造 |
| `encoder.py` | 蝶形 O(N log N) 编码 |
| `decoder_sc.py` | SC 译码 |
| `decoder_scl.py` | SCL / CA-SCL 译码 |
| `decoder_bp.py` | BP 译码（min-sum + 早停） |
| `simulation.py` | 蒙特卡洛仿真 |
| `utils.py` | 结果保存与绘图 |
| `results/` | CSV 与图像输出 |

## 约定说明

- LLR 符号：`LLR = ln P(y|0) / P(y|x=1) = 2y/σ²`
- 编码器输出含比特倒序置换；SC/SCL/BP 译码器内部对信道 LLR 做相应置换
- `frozen_bits[i]=1` 表示冻结位
