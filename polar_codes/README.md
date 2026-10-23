# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 快速开始

```bash
cd polar_codes
pip install -r requirements.txt
python validate.py
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

## 加速/调试环境变量

| 变量 | 含义 |
|------|------|
| `POLAR_MAX_FRAMES` | 每 SNR 点最大帧数（默认 100000） |
| `POLAR_MIN_ERRORS` | 每 SNR 点最少错误帧数（默认 100） |
| `POLAR_SKIP_N1024` | 设为 `1` 跳过 N=1024 |
| `POLAR_EB_MIN` / `POLAR_EB_MAX` / `POLAR_EB_STEP` | 仿真 Eb/N0 范围 |

完整实验在普通 CPU 上可能需要数小时；开发时可减小 `POLAR_MAX_FRAMES` 与 `POLAR_MIN_ERRORS`。
