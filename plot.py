import pandas as pd, matplotlib.pyplot as plt
df = pd.read_csv("results.csv")
order = ["1KB","10KB","100KB","1MB"]

plt.figure()
for m in df["method"].unique():
    d = df[df.method==m].set_index("slice").loc[order]
    plt.plot(order, d["BPB"], marker="o", label=m)
plt.xlabel("Text length"); plt.ylabel("BPB"); plt.legend(); plt.title("Length vs BPB")
plt.savefig("fig1_length_bpb.png", dpi=150)

plt.figure()
for m in df["method"].unique():
    d = df[df.method==m]
    plt.scatter(d["BPB"], d["throughput_MBps"], label=m)
plt.xlabel("BPB"); plt.ylabel("Throughput (MB/s)"); plt.legend(); plt.title("BPB vs Throughput")
plt.savefig("fig2_bpb_throughput.png", dpi=150)
print("图已保存：fig1_length_bpb.png, fig2_bpb_throughput.png")