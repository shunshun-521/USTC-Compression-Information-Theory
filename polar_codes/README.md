# 极化码（Polar Codes）编译码仿真

基于 Python（NumPy/SciPy）的极化码仿真工具链，不依赖第三方极化码库。

## 模块说明

| 文件 | 功能 |
|------|------|
| `construction.py` | 高斯近似（GA）极化码构造 |
| `encoder.py` | O(N log N) 蝶形编码 + 比特倒序 |
| `channel.py` | BPSK + AWGN 信道与 LLR |
| `decoder_sc.py` | SC 译码（非递归主实现） |
| `decoder_scl.py` | SCL / CA-SCL 译码（含 CRC-8/16） |
| `decoder_bp.py` | BP 译码（min-sum + 早停） |
| `simulation.py` | 蒙特卡洛仿真主循环 |
| `utils.py` | CSV 保存、BLER 绘图、香农限 |

## 运行实验

```bash
cd polar_codes
pip install -r requirements.txt
python3 run_exp1.py   # SC：N=256/512/1024
python3 run_exp2.py   # SCL / CA-SCL：N=512
python3 run_exp3.py   # BP vs SC/SCL：N=256/512
```

输出目录：`results/`（CSV、PNG/PDF 曲线、`frozen_sets.txt`）。

## 快速模式（自动化 / CI）

```bash
export POLAR_QUICK=1
export POLAR_MAX_FRAMES=2000   # 可选，默认 2000
export POLAR_MIN_ERRORS=20     # 可选，默认 20
python3 run_exp1.py
```

## 单元测试

各 `run_exp*.py` 启动时会自动执行编码器、SC 无损、L=1 SCL 等价性校验。
