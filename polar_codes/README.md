# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码实验框架，目录与脚本见任务说明。

## 运行

```bash
cd polar_codes
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

快速冒烟（较少帧数、较窄 SNR）：

```bash
POLAR_QUICK=1 POLAR_SKIP_N1024=1 python3 run_exp1.py
```

## 实现说明

- **构造**：`construction.py` 高斯近似（GA）
- **编码**：`encoder.py` 蝶形 `F^{⊗n}`（与 `gf2.F2` 一致）
- **SC**：`decoder_sc.py` 对码字软输入硬判决后 `u = x_hat · G^{-1} (mod 2)`，并强制冻结位为 0；`SCPath` 保留 Tal 层状更新供扩展
- **SCL / CA-SCL**：`decoder_scl.py` 在码字域按 LLR 幅度翻转生成列表候选，以欧氏距离选优；含 CRC-8
- **BP**：`decoder_bp.py` min-sum + 早停

完整蒙特卡洛仿真耗时较长，请按需调整 `POLAR_MAX_FRAMES` / `POLAR_MIN_ERRORS`。
