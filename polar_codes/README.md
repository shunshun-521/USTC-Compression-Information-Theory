# 极化码编译码仿真（Python / NumPy）

实现 GA 构造、SC / SCL / CA-SCL / BP 译码与蒙特卡洛仿真，无第三方极化码库。

## 环境

```bash
pip install -r requirements.txt
```

## 校验

```bash
python validate.py
```

## 实验

```bash
python run_exp1.py   # SC，N=256/512/1024
python run_exp2.py   # SCL / CA-SCL，N=512
python run_exp3.py   # SC / SCL / BP 对比
```

可选环境变量（加速调试）：`POLAR_MAX_FRAMES`、`POLAR_MIN_ERRORS`、`POLAR_EB_MIN`、`POLAR_EB_MAX`、`POLAR_EB_STEP`、`POLAR_SKIP_N1024`、`POLAR_N_ONLY`。

完整论文级曲线请使用默认 `max_frames=100000`、`min_errors=100` 重新跑实验；仓库内部分 CSV 可能来自较少帧数的试跑。

## 目录

见项目说明中的 `polar_codes/` 结构；结果输出在 `results/`。
