import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
import os

print("CWD:", os.getcwd())
print("backend:", matplotlib.get_backend())

df1 = pd.read_csv("results.csv")
df2 = pd.read_csv("results_llm.csv")
df  = pd.concat([df1, df2], ignore_index=True)
df.to_csv("results_merged.csv", index=False, encoding="utf-8-sig")
print("已合并:", len(df), "行")

order = ["1KB","10KB","100KB","1MB"]
df["slice"] = pd.Categorical(df["slice"], categories=order, ordered=True)

fig, ax = plt.subplots(figsize=(7,5))
for m in df["method"].unique():
    d = df[df.method==m].sort_values("slice").dropna(subset=["BPB"])
    ax.plot(d["slice"].astype(str), d["BPB"], marker="o", label=m,
            linewidth=2.5 if m=="GPTzip" else 1.5)
ax.set_xlabel("Text length"); ax.set_ylabel("BPB")
ax.set_title("Length vs BPB (Traditional + LLM)")
ax.legend(); ax.grid(alpha=0.3)
out1 = os.path.abspath("fig3_length_bpb_with_llm.png")
fig.savefig(out1, dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig3:", out1, "存在=", os.path.exists(out1), "大小=", os.path.getsize(out1) if os.path.exists(out1) else 0)

fig, ax = plt.subplots(figsize=(7,5))
for m in df["method"].unique():
    d = df[df.method==m]
    ax.scatter(d["BPB"], d["throughput_MBps"], label=m, s=80,
               marker="*" if m=="GPTzip" else "o")
ax.set_yscale("log")
ax.set_xlabel("BPB"); ax.set_ylabel("Throughput (MB/s, log)")
ax.set_title("BPB vs Throughput")
ax.legend(); ax.grid(alpha=0.3, which="both")
out2 = os.path.abspath("fig4_bpb_throughput_with_llm.png")
fig.savefig(out2, dpi=150, bbox_inches="tight")
plt.close(fig)
print("fig4:", out2, "存在=", os.path.exists(out2), "大小=", os.path.getsize(out2) if os.path.exists(out2) else 0)

print("DONE")
