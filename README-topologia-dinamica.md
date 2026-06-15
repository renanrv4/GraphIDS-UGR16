# Topologia Dinâmica no GraphIDS — Guia Rápido

> Feature streaming com grafo dinâmico, threshold adaptativo e cold start para IPs novos.

---

## Pré-requisitos

```bash
# Python 3.11 + uv
python3.11 --version
uv --version

# Instalar dependências (sem pyg-lib/torch-sparse — patch_pyg.py como fallback)
uv sync --python 3.11

# IMPORTANTE: sempre use --frozen nos comandos
```

## Dataset

```bash
# Baixar NF-UNSW-NB15-v3 (~577MB)
pip install kagglehub
python -c "
import kagglehub
p = kagglehub.dataset_download('annajumper/ids-datasets')
print(p)
"
mkdir -p data/NF-UNSW-NB15-v3
cp <caminho_do_download>/NF-UNSW-NB15-v3.csv data/NF-UNSW-NB15-v3/
```

## Testes (79 testes, todos devem passar)

```bash
uv run --python 3.11 --frozen python -m pytest tests/ -v
```

## Executar a Feature

### 1. Modo Batch (referência) — treino temporal com 30% dos dados

```bash
uv run --python 3.11 --frozen python run_training.py batch
```

Escreve progresso em `training_status.json`. Acompanhe com:
```bash
cat training_status.json                # status atual
tail -f training_runner.log             # logs completos
grep "EPOCH:" training_runner.log       # métricas por época
```

### 2. Modo Streaming (topologia dinâmica)

```bash
uv run --python 3.11 --frozen python run_training.py streaming
```

### 3. Batch + Streaming sequencial

```bash
uv run --python 3.11 --frozen python run_training.py both
```

## O Que Acontece no Streaming

```
CSV → StreamingNetFlowDataset (janelas de 1h, stride 30min)
        │
        ├── Fase 1 — Treino (60% janelas iniciais)
        │     Janela 0: treina modelo
        │     Janela 1: continua treinando
        │     ...
        │
        └── Fase 2 — Avaliação (40% janelas finais)
              Janela k:   avalia → AdaptiveThreshold → F1-score
              Janela k+1: avalia → threshold se ajusta
              ...
```

### Fluxo de Inferência (flow a flow)

```
1. DynamicGraph.add_edge(src_ip, dst_ip, features, timestamp)
2. DynamicGraph.get_current_graph()          → subgrafo ativo
3. SAGELayer.forward()                       → embedding da aresta
4. SlidingWindowBuffer.add(embedding)        → buffer deslizante
5. TransformerAE.forward(window)             → reconstrução
6. MSE(window, reconstruction)               → anomaly_score
7. AdaptiveThreshold.add_score(score, label) → threshold adaptativo
8. score > threshold ? ALERTA : OK
```

## Arquitetura dos Componentes

```
inference/streaming.py
├── SlidingWindowBuffer   — buffer circular de embeddings
├── AdaptiveThreshold     — MAD ou validation F1 + EMA
└── StreamingDetector     — orquestra o pipeline

utils/dynamic_graph.py
└── DynamicGraph          — grafo dinâmico com stale edge removal

models/graphids.py
├── SAGELayer             — com temporal_weights e node_mask
├── ColdStartNodeInitializer — neighbor_mean, default_embedding
└── GraphIDS              — encode_edges() com suporte streaming
```

## Hiperparâmetros do Runner (CPU, 8GB RAM)

| Parâmetro | Valor | Motivo |
|---|---|---|
| `--fraction` | 0.3 | 30% dos dados cabe em 8GB RAM |
| `--batch_size` | 8192 | Reduzido para CPU |
| `--num_epochs` | 30 | Suficiente para convergência |
| `--learning_rate` | 1e-4 | Do config ótimo do projeto |

Para alterar, edite `BASE_ARGS` em `run_training.py`.

## Tuning de Hiperparâmetros

```bash
uv run --python 3.11 --frozen python main.py \
  --dataset NF-UNSW-NB15-v3 --data_dir ./data \
  --fraction 0.3 --tune \
  --tune_space config_search_space/tuning_space.yaml \
  --tune_trials 30
```

Espaço de busca para streaming: `config_search_space/streaming_tuning_space.yaml`

## Problemas Conhecidos

| Problema | Causa | Status |
|---|---|---|
| `num_workers=0` obrigatório | Monkey-patch do neighbor_sample não é herdado por processos filho | Contornado |
| ~500s/época em CPU | `patch_pyg.py` é Python puro (vs C++ compilado) | Aceito para CPU |
| val_pr_auc oscila | Temporal split + fraction pequeno cria val set instável | Precisa de tuning |
| GLIBC warnings na inicialização | pyg-lib/torch-sparse exigem GLIBC >= 2.32 (sistema tem 2.31) | Inofensivo (patch_pyg substitui) |