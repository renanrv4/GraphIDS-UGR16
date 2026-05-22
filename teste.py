import pandas as pd

cols = [
    "timestamp",
    "duration",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "flags",
    "fwd",
    "tos",
    "packets",
    "bytes",
    "label"
]

df = pd.read_csv(
    "data/ugr16/march.week3.csv",
    header=None,
    names=cols,
    nrows=500000
)

print(df.head())
print(df["label"].value_counts())