# 极化码编译码仿真

Python（NumPy/SciPy）实现的极化码 GA 构造、SC/SCL/CA-SCL/BP 译码与蒙特卡洛仿真。

## 运行

```bash
cd polar_codes
pip install -r requirements.txt
python3 run_exp1.py
python3 run_exp2.py
python3 run_exp3.py
```

快速冒烟测试（减少帧数）：

```bash
POLAR_QUICK=1 python3 run_exp1.py
```

## 说明

- 编码与 SC/SCL 采用与标准 Kronecker `F` 变换一致的比特序（无额外比特倒序）。
- SC 主入口 `sc_decode` 调用递归参考实现；`sc_decode_nonrecursive` 为分层非递归结构（可继续优化）。
- BP 为因子图 min-sum 迭代译码；长码长下性能通常弱于 SC/SCL，可通过 `max_iter` 与 `alpha` 调节。
