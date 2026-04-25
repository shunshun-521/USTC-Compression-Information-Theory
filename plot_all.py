import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import os, sys

# ---- 1. 合并所有 csv ----
files = ["results.csv", "results_llm.csv",
         "results_GPTzip-medium.csv", "results_qwen.csv"]
dfs = []
for fp in files:
    if not os.path.exists(fp):
        print(f"[WARN] 缺失 {fp}")
        continue
    d = pd.read_csv(fp)
    print(f"[OK] {fp}: {len(d)} 行")
    dfs.append(d)
df = pd.concat(dfs, ignore_index=True)

# 把空字符串/未跑数据转成 NaN
for col in ["BPB","t_compress_s","t_decompress_s","throughput_MBps","comp_B"]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df.to_csv("results_all.csv", index=False, encoding="utf-8-sig")
print(f"\n合并后总行数: {len(df)}, 其中有效 BPB 行: {df['BPB'].notna().sum()}")

# ---- 2. 准备绘图样式 ----
order = ["1KB","10KB","100KB","1MB"]
df["slice"] = pd.Categorical(df["slice"], categories=order, ordered=True)
methods = ["gzip","bzip2","xz","zstd","GPTzip","GPTzip-medium","Qwen2.5-0.5B"]
# GPTzip 在 results_llm.csv 里就叫 GPTzip
colors = {
    "gzip":"#1f77b4", "bzip2":"#ff7f0e", "xz":"#2ca02c", "zstd":"#d62728",
    "GPTzip":"#9467bd", "GPTzip-medium":"#8c564b", "Qwen2.5-0.5B":"#e377c2"
}
markers = {
    "gzip":"o", "bzip2":"o", "xz":"o", "zstd":"o",
    "GPTzip":"*", "GPTzip-medium":"*", "Qwen2.5-0.5B":"*"
}

# ---- 图 5：长度 - BPB（含三种 LLM）----
plt.figure(figsize=(8,5.5))
for m in methods:
    d = df[df.method==m].dropna(subset=["BPB"]).sort_values("slice")
    if len(d)==0: continue
    plt.plot(d["slice"].astype(str), d["BPB"],
             marker=markers[m], label=m, color=colors[m],
             linewidth=2.5 if m.startswith(("GPTzip","Qwen")) else 1.5,
             markersize=12 if markers[m]=="*" else 8)
plt.xlabel("Text length"); plt.ylabel("BPB")
plt.title("Length vs BPB: 4 Traditional + 3 LLM Compressors")
plt.legend(loc="upper right", fontsize=9)
plt.grid(alpha=0.3)
plt.savefig("fig5_length_bpb_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("写入 fig5_length_bpb_all.png")

# ---- 图 6：BPB - 吞吐量（log 轴）----
plt.figure(figsize=(8,5.5))
for m in methods:
    d = df[df.method==m].dropna(subset=["BPB","throughput_MBps"])
    if len(d)==0: continue
    plt.scatter(d["BPB"], d["throughput_MBps"],
                marker=markers[m], label=m, color=colors[m],
                s=150 if markers[m]=="*" else 80,
                edgecolors="black", linewidths=0.5)
plt.yscale("log")
plt.xlabel("BPB"); plt.ylabel("Throughput (MB/s, log)")
plt.title("Compression Ratio vs Throughput")
plt.legend(loc="lower left", fontsize=9)
plt.grid(alpha=0.3, which="both")
plt.savefig("fig6_bpb_throughput_all.png", dpi=150, bbox_inches="tight")
plt.close()
print("写入 fig6_bpb_throughput_all.png")

# ---- 图 7：三种 LLM 在各切片上的 BPB 对比（柱状图）----
fig, ax = plt.subplots(figsize=(8,5))
llm_methods = ["GPTzip","GPTzip-medium","Qwen2.5-0.5B"]
slices_for_bar = ["1KB","10KB","100KB","1MB"]
import numpy as np
x = np.arange(len(slices_for_bar))
w = 0.25
for i, m in enumerate(llm_methods):
    bpbs = []
    for s in slices_for_bar:
        v = df[(df.method==m) & (df["slice"].astype(str)==s)]["BPB"].values
        bpbs.append(v[0] if len(v)>0 and not pd.isna(v[0]) else None)
    # 把 None 替换成 0 但用 hatch 标记
    heights = [b if b is not None else 0 for b in bpbs]
    bars = ax.bar(x + (i-1)*w, heights, w, label=m, color=colors[m],
                  edgecolor="black", linewidth=0.5)
    # 在没数据的柱子上写"未做"
    for b, val, bar in zip(bpbs, heights, bars):
        if b is None:
            ax.text(bar.get_x()+bar.get_width()/2, 0.05, "N/A",
                    ha="center", fontsize=8, color="gray")
        else:
            ax.text(bar.get_x()+bar.get_width()/2, val+0.03,
                    f"{val:.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(slices_for_bar)
ax.set_xlabel("Text length"); ax.set_ylabel("BPB")
ax.set_title("Three LLM Compressors: BPB Comparison")
ax.legend(); ax.grid(alpha=0.3, axis="y")
plt.savefig("fig7_llm_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("写入 fig7_llm_comparison.png")

# ---- 简短汇总 ----
print("\n=== 各切片上 BPB 最低的方法 ===")
for s in slices_for_bar:
    d = df[df["slice"].astype(str)==s].dropna(subset=["BPB"])
    if len(d)==0: continue
    best = d.loc[d["BPB"].idxmin()]
    print(f"  {s:6s}: {best['method']:15s} BPB={best['BPB']:.4f}")
print("\n所有图已生成，CSV 已合并到 results_all.csv")
