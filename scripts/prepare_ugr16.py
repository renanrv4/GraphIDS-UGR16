import pandas as pd

# Colunas originais do UGR16
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

print("Lendo dataset...")

df = pd.read_csv(
    "data/ugr16/march.week3.csv",
    header=None,
    names=cols,
    nrows=1_000_000
)

print("Convertendo timestamps...")

# Timestamp -> epoch milliseconds
df["FLOW_START_MILLISECONDS"] = (
    pd.to_datetime(df["timestamp"]).astype("int64") // 10**6
)

print("Convertendo protocolos...")

proto_map = {
    "ICMP": 1,
    "TCP": 6,
    "UDP": 17
}

df["PROTOCOL"] = df["protocol"].map(proto_map)

print("Convertendo flags...")

# Encoding simples para flags
df["TCP_FLAGS"] = pd.factorize(df["flags"])[0]

print("Criando labels...")

# Attack textual
df["Attack"] = df["label"]

# Label binário
df["Label"] = (df["label"] != "background").astype(int)

print("Montando dataframe final...")

final_df = pd.DataFrame({
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
    "Label": df["Label"]
})

print(final_df.head())

print("Salvando dataset convertido...")

final_df.to_csv(
    "data/UGR16-v3/UGR16-v3.csv",
    index=False
)

print("Concluído!")