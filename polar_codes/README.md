# 极化码（Polar Codes）编译码仿真

Python（NumPy/SciPy）实现的极化码全流程：GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛 BLER 仿真。**不依赖**第三方极化码库。

## 目录

| 文件 | 说明 |
|------|------|
| `construction.py` | 高斯近似（GA）极化码构造 |
| `encoder.py` | O(N log N) 蝶形编码 + 比特倒序 |
| `channel.py` | BPSK + AWGN + LLR |
| `decoder_sc.py` | SC（递归 + 非递归） |
| `decoder_scl.py` | SCL / CA-SCL（CRC-8/16） |
| `decoder_bp.py` | BP（min-sum + 早停） |
| `simulation.py` | 蒙特卡洛主循环 |
| `utils.py` | CSV、绘图、香农限 |
| `run_exp1.py` ~ `run_exp3.py` | 三组实验 |
| `validate.py` | 单元与一致性校验 |

## 快速校验

```bash
cd polar_codes
pip install -r requirements.txt
python3 validate.py
```

编码器约定：`u=[1,0,1,1]` → `x=[1,0,1,1]`（`G_N = B_N F^{⊗n}`，含比特倒序）。

GA 核对：`N=8,K=4,Eb/N0=2.5dB` → `info=[0,3,5,6]`；`N=256` 前 20 个信息位索引见 `python3 construction.py`。

## 运行实验

完整蒙特卡洛（每 SNR 点最多 10⁵ 帧、至少 100 误帧）耗时较长，可用环境变量缩小规模做冒烟：

| 变量 | 默认 | 说明 |
|------|------|------|
| `POLAR_MAX_FRAMES` | 100000 | 每 Eb/N0 最大帧数 |
| `POLAR_MIN_ERRORS` | 100 | 最少误帧数 |
| `POLAR_EB_MIN` / `MAX` / `STEP` | 0~5.5, 0.25 | Eb/N0 扫描 |
| `POLAR_SKIP_N1024` | 0 | 设为 `1` 跳过 exp1 的 N=1024 |

```bash
python3 run_exp1.py   # SC: N=256,512,1024
python3 run_exp2.py   # SCL / CA-SCL: N=512
python3 run_exp3.py   # SC vs SCL vs BP
```

输出在 `results/`：`*.csv`、`frozen_sets.txt`、`fig*.png` / `*.pdf`。
