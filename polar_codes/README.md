# 极化码编译码仿真（Python）

## 依赖

```bash
pip install -r requirements.txt
```

## 校验

```bash
python validate.py
```

## 实验

```bash
python run_exp1.py
python run_exp2.py
python run_exp3.py
```

结果输出至 `results/`。

## 说明

- 信道 LLR 进入译码器前需做比特倒序重排：`channel.channel_llr_to_decoder`。
- `sc_decode` 采用因子图单次双向消息传递（与 BP 单次迭代等价，min-sum f 运算）。
- 编码器校验向量：`u=[1,0,1,1] -> x=[1,0,1,1]`（$G_N=B_N F^{\otimes n}$ 约定）。
