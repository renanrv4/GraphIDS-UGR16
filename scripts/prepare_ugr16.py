import pandas as pd

# ==========================
# Original columns of the UGR16 dataset
# ==========================

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
    "label",
]

# Using only 1000000 rows for faster processing, but you can adjust this as needed
# UGR16: https://nesg.ugr.es/nesg-ugr16/
df = pd.read_csv("data/ugr16/march.week3.csv", header=None, names=cols, nrows=1_000_000)

print("Converting timestamps...")

# ==========================
# Timestamp -> epoch milliseconds
# ==========================

df["FLOW_START_MILLISECONDS"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**6

# ==========================
# Mapping of protocols to integers (ICMP=1, TCP=6, UDP=17)
# ==========================

print("Converting protocols...")

proto_map = {"ICMP": 1, "TCP": 6, "UDP": 17}

df["PROTOCOL"] = df["protocol"].map(proto_map)


# ==========================
# Encoding of TCP flags (treating as categorical)
# ==========================

print("Converting flags...")

df["TCP_FLAGS"] = pd.factorize(df["flags"])[0]

# ==========================
# Attack labels (keeping original string labels for reference)
# ==========================

print("Creating labels...")
df["Attack"] = df["label"]

# ==========================
# Label binário
# ==========================

df["Label"] = (df["label"] != "background").astype(int)


# ==========================
# Final DataFrame with selected and renamed columns
# ==========================

print("Final DataFrame...")

final_df = pd.DataFrame(
    {
        "IPV4_SRC_ADDR": df["src_ip"],
        "IPV4_DST_ADDR": df["dst_ip"],
        "FLOW_START_MILLISECONDS": df["FLOW_START_MILLISECONDS"],
        "DURATION": df["duration"],
        "L4_SRC_PORT": df["src_port"],
        "L4_DST_PORT": df["dst_port"],
        "PROTOCOL": df["PROTOCOL"],
        "TCP_FLAGS": df["TCP_FLAGS"],
        "FWD": df["fwd"],
        "TOS": df["tos"],
        "IN_PKTS": df["packets"],
        "IN_BYTES": df["bytes"],
        "Attack": df["Attack"],
        "Label": df["Label"],
    }
)

print(final_df.head())

# ==========================
# SAVE
# ==========================

print("Saving dataset...")

final_df.to_csv("data/UGR16-v3/UGR16-v3.csv", index=False)

print("Done")
print("--------------------------------")
