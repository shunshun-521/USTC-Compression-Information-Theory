# 极化码（Polar Codes）编译码仿真

Python（NumPy/SciPy）实现，无第三方极化码库。包含 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
cd polar_codes
python3 validate.py
```

## 实验

```bash
python3 run_exp1.py   # SC：N=256/512/1024
python3 run_exp2.py   # SCL / CA-SCL：N=512
python3 run_exp3.py   # SC vs SCL vs BP：N=256/512
```

可通过环境变量缩短仿真（CI/调试）：

| 变量 | 含义 |
|------|------|
| `POLAR_MAX_FRAMES` | 每信噪比点最大帧数（默认 100000） |
| `POLAR_MIN_ERRORS` | 每点最少错误帧数（默认 100） |
| `POLAR_EB_MIN` / `POLAR_EB_MAX` / `POLAR_EB_STEP` | Eb/N0 扫描范围 |
| `POLAR_SKIP_N1024` | 设为 `1` 跳过实验一 N=1024 |
| `POLAR_N256_ONLY` | 设为 `1` 时实验三仅跑 N=256 |

结果输出至 `results/`（CSV、PNG/PDF 曲线、`frozen_sets.txt`）。
