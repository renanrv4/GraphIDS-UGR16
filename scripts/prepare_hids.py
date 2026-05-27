import os
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

print("Lendo dataset (Parquet via PyArrow / row groups)...")

INPUT_PARQUET = "data/process_uber_summary/process_uber_summary.parquet"
OUTPUT_CSV = "data/ACME-HIDS-v1/ACME-HIDS-v1.csv"
NROWS = 2_000_000  # ajuste aqui (ou None para ler tudo)

# Só as colunas que realmente usamos no preprocessing abaixo
# (ler menos colunas = muito menos RAM)
NEEDED_COLS = [
    "process_started_seconds",
    "duration_seconds",
    "pid_hash",
    "parent_pid_hash",
    "label_num_hits",
    "tcp_connect_count",
    "tcp_send_count",
    "tcp_recv_count",
    "udp_send_count",
    "udp_recv_count",
    "net_send_vs_recv",
    "cpu_utilization",
    "net_total_events",
    "net_total_size",
    "Read_Events",
    "Write_Events",
    "Create_Events",
    "Delete_Events",
    "Rename_Events",
    "reg_totals",
    "Read_Bytes",
    "Write_Bytes",
]

pf = pq.ParquetFile(INPUT_PARQUET)

tables = []
read_rows = 0
target = NROWS if NROWS is not None else float("inf")

for rg in range(pf.num_row_groups):
    # Tenta ler todas as colunas necessárias.
    # Se alguma não existir (ex.: token_elevation_type), caímos no fallback.
    try:
        t = pf.read_row_group(rg, columns=NEEDED_COLS)
    except Exception:
        # fallback: lê as colunas que existirem de fato
        schema_names = set(pf.schema.names)
        existing_cols = [c for c in NEEDED_COLS if c in schema_names]
        t = pf.read_row_group(rg, columns=existing_cols)

    tables.append(t)
    read_rows += t.num_rows
    if read_rows >= target:
        break

# Converte para pandas e corta exatamente NROWS
df = pd.concat([t.to_pandas() for t in tables], ignore_index=True)
if NROWS is not None:
    df = df.head(NROWS).copy()

print("Linhas/colunas carregadas:", df.shape)

print("Convertendo timestamps...")

# Garante numérico e remove inf
df["process_started_seconds"] = pd.to_numeric(df["process_started_seconds"], errors="coerce")
df["process_started_seconds"] = df["process_started_seconds"].replace([np.inf, -np.inf], np.nan)

# Remove linhas sem timestamp (necessário para o modelo)
before = len(df)
df = df.dropna(subset=["process_started_seconds"]).copy()
after = len(df)
print(f"Removidas {before - after} linhas sem process_started_seconds")

# Agora sim pode virar int
df["FLOW_START_MILLISECONDS"] = (df["process_started_seconds"].astype("int64") * 1000)

df["DURATION"] = pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0).astype("float32")
print("Montando grafo processo->processo (parent -> child)...")

df["IPV4_SRC_ADDR"] = df["parent_pid_hash"].astype(str)
df.loc[df["parent_pid_hash"].isna(), "IPV4_SRC_ADDR"] = "ROOT"
df["IPV4_DST_ADDR"] = df["pid_hash"].astype(str)

print("Criando features compatíveis...")

# PROTOCOL (adaptado): 6 se tiver TCP, 17 se tiver UDP, senão 0
for c in ["tcp_connect_count", "tcp_send_count", "tcp_recv_count", "udp_send_count", "udp_recv_count"]:
    if c not in df.columns:
        df[c] = 0

tcp_activity = df["tcp_connect_count"].fillna(0) + df["tcp_send_count"].fillna(0) + df["tcp_recv_count"].fillna(0)
udp_activity = df["udp_send_count"].fillna(0) + df["udp_recv_count"].fillna(0)

df["PROTOCOL"] = np.select([tcp_activity > 0, udp_activity > 0], [6, 17], default=0).astype("int16")

df["L4_SRC_PORT"] = 0
df["L4_DST_PORT"] = 0

# "TCP_FLAGS" (adaptado): encoding de token_elevation_type, se existir
if "token_elevation_type" in df.columns:
    df["TCP_FLAGS"] = pd.factorize(df["token_elevation_type"].astype(str))[0].astype("int32")
else:
    df["TCP_FLAGS"] = -1

# FWD / TOS
if "net_send_vs_recv" not in df.columns:
    df["net_send_vs_recv"] = 0
df["FWD"] = (
    df["net_send_vs_recv"]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0)
    .astype("float32")
)

if "cpu_utilization" not in df.columns:
    df["cpu_utilization"] = 0
df["TOS"] = df["cpu_utilization"].fillna(0).astype("float32")

# IN_PKTS / IN_BYTES: proxies
for c in [
    "net_total_events", "Read_Events", "Write_Events", "Create_Events", "Delete_Events", "Rename_Events", "reg_totals",
    "net_total_size", "Read_Bytes", "Write_Bytes",
]:
    if c not in df.columns:
        df[c] = 0

df["IN_PKTS"] = (
    df["net_total_events"].fillna(0)
    + df["Read_Events"].fillna(0)
    + df["Write_Events"].fillna(0)
    + df["Create_Events"].fillna(0)
    + df["Delete_Events"].fillna(0)
    + df["Rename_Events"].fillna(0)
    + df["reg_totals"].fillna(0)
).astype("float32")

df["IN_BYTES"] = (
    df["net_total_size"].fillna(0)
    + df["Read_Bytes"].fillna(0)
    + df["Write_Bytes"].fillna(0)
).astype("float32")

print("Criando labels...")

if "label_num_hits" not in df.columns:
    raise ValueError("Coluna 'label_num_hits' não foi lida/encontrada no parquet.")

df["Attack"] = np.where(df["label_num_hits"].fillna(0) > 0, "labeled_attack", "benign")
df["Label"] = (df["label_num_hits"].fillna(0) > 0).astype(int)

print("Montando dataframe final...")

final_df = pd.DataFrame(
    {
        "IPV4_SRC_ADDR": df["IPV4_SRC_ADDR"],
        "IPV4_DST_ADDR": df["IPV4_DST_ADDR"],
        "FLOW_START_MILLISECONDS": df["FLOW_START_MILLISECONDS"],
        "DURATION": df["DURATION"],
        "L4_SRC_PORT": df["L4_SRC_PORT"],
        "L4_DST_PORT": df["L4_DST_PORT"],
        "PROTOCOL": df["PROTOCOL"],
        "TCP_FLAGS": df["TCP_FLAGS"],
        "FWD": df["FWD"],
        "TOS": df["TOS"],
        "IN_PKTS": df["IN_PKTS"],
        "IN_BYTES": df["IN_BYTES"],
        "Attack": df["Attack"],
        "Label": df["Label"],
    }
)

print(final_df.head())

print("Salvando dataset convertido...")

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
final_df.to_csv(OUTPUT_CSV, index=False)

print("Concluído!")