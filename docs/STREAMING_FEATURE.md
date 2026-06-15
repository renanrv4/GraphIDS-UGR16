# Feature de Streaming para Detecção de Anomalias em Grafos de Fluxo de Rede

> **Compatibilidade**: O código inclui `patch_pyg.py` que implementa um fallback em Python puro para o `NeighborSampler` do PyG, eliminando a dependência de `pyg-lib` e `torch-sparse` (que exigem GLIBC >= 2.32). Funciona em sistemas como Ubuntu 20.04 com GLIBC 2.31.

## 1. Problema

O artigo original do **GraphIDS** propõe um modelo híbrido GNN + Transformer Autoencoder para detecção de anomalias em fluxos de rede. O pipeline original opera em **modo batch**: todo o dataset é carregado em memória, particionado em treino/validação/teste (split estratificado ou temporal), e o modelo treina por épovas sobre o grafo estático completo.

Isso apresenta três limitações fundamentais para cenários reais de NIDS (Network Intrusion Detection):

1. **Não adaptabilidade a conceitos novos (concept drift)**: Distribuições de tráfego mudam ao longo do tempo. Um modelo treinado uma única vez perde performance quando o padrão de tráfego muda (ex.: novos protocolos, novos dispositivos, mudança de horário/dia).

2. **Impossibilidade de inferência contínua**: O pipeline batch não suporta a chegada de novos fluxos em tempo real. Cada nova inferência exigiria reprocessar todo o dataset.

3. **Estalecimento do grafo (graph staleness)**: O grafo completo contém arestas antigas que podem não refletir mais o estado atual da rede. Manter arestas obsoletas degrada a qualidade das embeddings geradas pelo encoder GNN.

4. **Cold start para novos nós**: Quando um novo IP aparece na rede, ele não possui embedding no grafo. O modelo original simplesmente ignora nós novos, o que em cenário streaming é inaceitável.

5. **Threshold fixo**: O threshold de anomalia é calculado uma única vez no validation set e reutilizado para sempre. Em cenário streaming, o limiar ideal muda conforme a distribuição dos erros de reconstrução evolui.

---

## 2. Solução Proposta

Implementamos uma **arquitetura de streaming** que estende o GraphIDS com:

```mermaid
flowchart LR
    A[Fluxos de Rede] --> B[DynamicGraph]
    B --> C[SAGELayer Encoder]
    C --> D[SlidingWindowBuffer]
    D --> E[Transformer AE]
    E --> F[AdaptiveThreshold]
    F --> G[Alerta / Score]

    B -.-> H[Edge Removal<br/>(stale edges)]
    C -.-> I[Temporal Weights]
    C -.-> J[Node Mask / ColdStart]
```

### 2.1 Dynamic Graph (`utils/dynamic_graph.py`)

Grafo dinâmico que evolui com a chegada de novos fluxos:

- **Mapeamento IP → node_id**: Cria nós sob demanda. Se um IP nunca foi visto, um novo ID é gerado automaticamente. Limite configurável de `max_nodes`.
- **Adição de arestas**: `add_edge()` e `add_edges_batch()` inserem novas arestas com seus atributos (features, timestamp, label).
- **Remoção de arestas obsoletas**: `remove_stale_edges()` remove arestas mais velhas que `max_edge_age_ms`. Mantém o grafo enxuto e relevante.
- **Remoção de nós inativos**: `remove_inactive_nodes()` remove nós que não tiveram atividade recente.
- **Snapshots temporais**: `get_current_graph()` retorna o estado atual; `get_snapshot(start, end)` retorna um subgrafo temporal; `get_snapshot_dataframe()` retorna como DataFrame para depuração.
- **Scaler online**: `fit_scaler()`, `transform_features()`, `inverse_transform_features()` para normalização consistente.

### 2.2 StreamingDetector (`inference/streaming.py`)

Orquestra o pipeline de inferência contínua:

#### SlidingWindowBuffer
- Buffer circular thread-safe (`deque(maxlen=window_size)`)
- `add(embedding)`: insere embedding do encoder
- `get_window()`: retorna tensor `[1, window_size, dim]` + máscara booleana indicando posições preenchidas
- Eficiente para memória: embeddings são mantidos em CPU, transferidos para GPU apenas na inferência

#### AdaptiveThreshold
- **MAD (Median Absolute Deviation)**: `threshold = median + multiplier * MAD`. Robusto a outliers. Se MAD=0, fallback para std.
- **Validation F1**: Busca o threshold que maximiza F1-score sobre o histórico de erros com labels conhecidas. Varre percentis entre 50% e 99.9%.
- `add_score(score, label)`: acumula histórico (últimos `window_size` scores)
- `compute()`: recalcula threshold com base no histórico acumulado
- Thread-safe para processamento concorrente

#### StreamingDetector
- `process_flow()`: processa um fluxo individual
  1. Adiciona aresta ao DynamicGraph
  2. Obtém snapshot do grafo atual
  3. Executa encoder → embedding
  4. Adiciona embedding ao buffer
  5. Se buffer pronto, executa Transformer AE → erro de reconstrução
  6. Atualiza threshold adaptativo
  7. Retorna (anomaly_score, is_anomalous)
- `process_batch()`: versão vetorizada para lote de fluxos
- `get_statistics()`: métricas operacionais (total_flows, total_alerts, alert_rate, current_threshold, buffer_fill)
- `reset()`: limpa todo o estado interno

### 2.3 Adaptações no Modelo (`models/graphids.py`)

#### SAGELayer com Temporal Weights e Node Mask

```python
def forward(self, edge_index, edge_attr, edge_couples, num_nodes,
            temporal_weights=None, node_mask=None):
```

- **`temporal_weights`**: Tensor de pesos por aresta (ex.: decay exponencial baseado na idade). A aresta mais recente tem peso 1.0; arestas antigas têm peso reduzido. Aplicado no método `message()`:
  ```python
  def message(self, edge_attr, edge_temporal_weights=None):
      if edge_temporal_weights is not None:
          return edge_attr * edge_temporal_weights.unsqueeze(-1)
      return edge_attr
  ```
  Isso faz com que arestas recentes contribuam mais para a embedding do nó.

- **`node_mask`**: Máscara de nós a considerar. Arestas incidentes a nós fora da máscara são filtradas antes da propagação. Permite inferência sobre subgrafos.

- **`torch.nan_to_num`**: Trata nós sem arestas incidentes (grau zero), que produziriam NaN na agregação. Esses nós recebem embedding zero.

#### ColdStartNodeInitializer

```python
class ColdStartNodeInitializer:
    def __init__(self, ndim: int, strategy: str = "neighbor_mean"):
```

Estratégias:
1. **`neighbor_mean`** (padrão): Para nós novos sem embedding, calcula a média das embeddings dos vizinhos (1-hop). Se não houver vizinhos válidos:
2. **`default_embedding`**: Usa um tensor aprendível como embedding padrão para nós desconhecidos
3. **Fallback**: Vetor de zeros

#### GraphIDS.encode_edges()

Nova API pública que recebe `DynamicGraphData` diretamente:
```python
def encode_edges(self, dynamic_graph_data: Data,
                 edge_couples: torch.Tensor,
                 temporal_weights=None) -> torch.Tensor:
```

### 2.4 Streaming Dataset (`utils/dataloaders.py`)

#### StreamingNetFlowDataset

Processa o CSV completo em **janelas temporais deslizantes** (sliding windows):

```
window_size_ms = 3600000  (1 hora)
stride_ms      = 1800000  (30 minutos)
```

- Ordena fluxos por timestamp
- Para cada janela `[t, t + window_size)`:
  1. Adiciona fluxos ao DynamicGraph
  2. Extrai snapshot `get_snapshot(t, t + window_size)`
  3. Armazena como `Data` object (edge_index, edge_attr, edge_labels, node_features)
- Propriedades:
  - `num_windows`: total de janelas temporais
  - `get_window(i)`: retorna o `Data` da i-ésima janela
  - `add_flows(df)`: adiciona novos fluxos e gera novas janelas incrementalmente

O treino em modo streaming funciona em duas fases:

1. **Fase de treino** (primeiros 60% das janelas):
   ```python
   for i in range(split_idx):
       window_data = dataset.get_window(i)
       loader = LinkNeighborLoader(data=window_data, ...)
       model, _, _ = train(model, loader, loader, ...)
   ```
   Treina o modelo janela por janela, cada uma por `max(5, num_epochs // num_windows)` épocas.

2. **Fase de avaliação** (40% restantes):
   ```python
   for i in range(split_idx, num_windows):
       window_data = dataset.get_window(i)
       val_loader = LinkNeighborLoader(data=window_data, ...)
       _, errors, labels = validate(model, val_loader, ...)
       threshold = update_threshold_online(errors, labels,
                                           method="validation_f1",
                                           prev_threshold=threshold)
   ```
   Avalia janela a janela com threshold adaptativo via **exponential moving average**:
   ```python
   def update_threshold_online(errors, labels, method, multiplier,
                                prev_threshold, alpha=0.1):
       computed = find_threshold(errors, labels, method, multiplier)
       if prev_threshold is not None:
           return alpha * computed + (1 - alpha) * prev_threshold
       return computed
   ```
   O parâmetro `alpha=0.1` dá peso 10% ao threshold da janela atual e 90% ao histórico, suavizando transições.

### 2.5 Busca de Hiperparâmetros (`main.py`)

O pipeline de **tuning** suporta busca aleatória sobre espaço definido em YAML:

```python
def tune_hyperparameters(args, dataset, base_config_dict):
    space = yaml.safe_load(open(args.tune_space))
    for trial in range(args.tune_trials):
        overrides = _sample_from_space(space, rng)
        # train + validate, registra val_pr_auc
        # tracking com wandb (offline por padrão)
```

Dois espaços de busca:
- `config_search_space/tuning_space.yaml`: Batch mode (7 hiperparâmetros)
- `config_search_space/streaming_tuning_space.yaml`: Streaming mode (10 hiperparâmetros, inclui `window_size`, `batch_size`, `step_percent`, `dropout`)

A amostragem suporta `loguniform`, `uniform`, `int`, `choice`.

---

## 3. Implementação Detalhada

### 3.1 Arquitetura de Componentes

```
main.py
├── Modo Batch (default)
│   ├── NetFlowDataset → train/val/test graphs estáticos
│   ├── train_model(): treina GNN encoder + Transformer AE
│   └── test(): avalia no test set
│
├── Modo Streaming (--streaming)
│   ├── StreamingNetFlowDataset → janelas temporais
│   ├── run_streaming():
│   │   ├── Fase 1: treino janela a janela (60%)
│   │   └── Fase 2: avaliação com threshold adaptativo (40%)
│
└── Modo Tuning (--tune)
    └── tune_hyperparameters(): busca aleatória + wandb
```

```
inference/streaming.py
├── SlidingWindowBuffer
├── AdaptiveThreshold
└── StreamingDetector
    ├── process_flow()
    ├── process_batch()
    └── get_statistics()
```

```
utils/dynamic_graph.py
└── DynamicGraph
    ├── add_edge() / add_edges_batch()
    ├── remove_stale_edges()
    ├── remove_inactive_nodes()
    ├── get_current_graph() / get_snapshot()
    └── fit_scaler() / transform_features()
```

```
models/graphids.py
├── SAGELayer (+ temporal_weights, node_mask)
├── ColdStartNodeInitializer
└── GraphIDS
    ├── encode_edges()
    ├── save_checkpoint()
    └── load_checkpoint()
```

### 3.2 Fluxo de Dados (Streaming, flow a flow)

```
Fluxo de rede {src_ip, dst_ip, features, timestamp}
    │
    ▼
1. DynamicGraph.add_edge(src_ip, dst_ip, features, timestamp)
    │
    ▼
2. DynamicGraph.get_current_graph()
    │  └── Subgrafo com nós e arestas ativas
    ▼
3. SAGELayer.forward(edge_index, edge_attr, edge_couples, num_nodes)
    │  └── embeddings das arestas de interesse
    ▼
4. SlidingWindowBuffer.add(embedding)
    │
    ▼
5. SlidingWindowBuffer.get_window()
    │  └── [1, window_size, embedding_dim] + mask
    ▼
6. TransformerAutoencoder.forward(window, padding_mask)
    │  └── reconstrução
    ▼
7. MSE(window, reconstruction) → anomaly_score
    │
    ▼
8. AdaptiveThreshold.add_score(score, label)
   AdaptiveThreshold.compute() → threshold
    │
    ▼
9. score > threshold ? ALERTA : OK
```

---

## 4. Como Testamos

### 4.1 Testes de DynamicGraph (45 testes)

Arquivo: `tests/test_dynamic_graph.py`

| Categoria | Testes | O que cobre |
|-----------|--------|-------------|
| Node Mapping | 5 | Criação de IDs, reuso por IP, limite max_nodes, consistência inverse_map |
| Edge Addition | 7 | IDs sequenciais, counts, batch, atributos, timestamps, labels |
| Edge Removal | 5 | Marcação inactive, retorno False para ID inválido, stale edges |
| Node Removal | 5 | Remoção de arestas incidentes, remoção de node_map, stale nodes |
| Snapshot | 7 | Shapes corretos, empty graph, filtro temporal, features all-ones, DataFrame |
| Query Methods | 7 | num_nodes, num_edges, get_node_ids, get_edge_ids_for_node, degree |
| Reset | 2 | Clear total, counts zero pós-reset |
| Scaler | 5 | Fit, transform, inverse transform, erro sem fit |

### 4.2 Testes de Streaming (22 testes)

Arquivo: `tests/test_streaming.py`

| Classe | Testes | O que cobre |
|--------|--------|-------------|
| SlidingWindowBuffer | 7 | Ready state, shape do window, máscara, wrap no overflow, clear |
| AdaptiveThreshold | 8 | MAD method, MAD=0 fallback, validation_f1, reset, thread safety |
| StreamingDetector | 7 | process_flow com mock, empty graph, batch processing, statistics, reset |

### 4.3 Testes do Modelo Adaptado (12 testes)

Arquivo: `tests/test_model_adaptations.py`

| Classe | Testes | O que cobre |
|--------|--------|-------------|
| SAGELayer | 5 | temporal_weights, node_mask, NaN handling, backward compat |
| ColdStartNodeInitializer | 3 | neighbor_mean, default_embedding, preservação de não-NaN |
| GraphIDS | 4 | encode_edges, forward com/sem novos parâmetros |

**Total: 79 testes — todos passando.**

---

## 5. Aprimoramentos em Relação ao Artigo Original

| Aspecto | Original (Batch) | Novo (Streaming) |
|---------|-----------------|-------------------|
| **Grafo** | Estático, particionado uma vez | Dinâmico, evolui com cada fluxo |
| **Treinamento** | Épocas sobre dataset completo | Janelas temporais deslizantes |
| **Inferência** | Batch sobre test set estático | Flow a flow, tempo real |
| **Adaptação** | Nenhuma (modelo congelado) | Threshold adaptativo (MAD/F1 + EMA) |
| **Staleness** | Arestas antigas sempre presentes | `remove_stale_edges()` automático |
| **Novos nós** | Ignorados ou causam erro | `ColdStartNodeInitializer` (vizinho média ou embedding padrão) |
| **Pesos temporais** | Não existem | `temporal_weights` no SAGELayer |
| **Threshold** | Fixo, calculado no val set | Adaptativo online (MAD ou validation F1) |
| **Buffer** | Não existe | `SlidingWindowBuffer` para sequência de embeddings |
| **Dataset** | `NetFlowDataset` (3 splits) | `StreamingNetFlowDataset` (N janelas) |
| **Tuning** | Manual | Automatizado com busca aleatória + wandb |
| **Pipeline** | Script único | `train.sh` + `tune.sh` para reprodução |

### 5.1 Impacto Esperado

1. **Detecção em tempo real**: Cada fluxo é processado individualmente com latência de milissegundos (apenas forward pass do modelo).
2. **Adaptação a concept drift**: O threshold se ajusta continuamente. O modelo pode ser re-treinado periodicamente com novas janelas.
3. **Grafo sempre relevante**: Arestas com mais de `max_edge_age_ms` são removidas automaticamente.
4. **Robustez a novos IPs**: `ColdStartNodeInitializer` garante que nós nunca antes vistos recebam embeddings plausíveis via agregação de vizinhos.
5. **Reprodutibilidade**: Scripts `train.sh` e `tune.sh` padronizados e testados.
6. **Espaço de busca específico**: `streaming_tuning_space.yaml` com hiperparâmetros (window_size=64-256, batch_size=8192-32768) otimizados para modo streaming.

---

## 6. Estrutura de Arquivos (Relevante)

```
.
├── main.py                              # Entry point (batch, streaming, tuning)
├── models/
│   └── graphids.py                      # SAGELayer, ColdStartNodeInitializer, GraphIDS
├── utils/
│   ├── dynamic_graph.py                 # DynamicGraph (core do grafo dinâmico)
│   ├── dataloaders.py                   # StreamingNetFlowDataset
│   └── trainers.py                      # train, validate, test, train_online, update_threshold_online
├── inference/
│   └── streaming.py                     # SlidingWindowBuffer, AdaptiveThreshold, StreamingDetector
├── patch_pyg.py                         # Fallback PyTorch puro para NeighborSampler (GLIBC 2.31)
├── run_training.py                      # Runner com status file e métricas periódicas
├── pipelines/
│   ├── train.sh                         # Pipeline batch + streaming
│   └── tune.sh                          # Pipeline de tuning
├── config_search_space/
│   ├── tuning_space.yaml                # Batch tuning
│   └── streaming_tuning_space.yaml      # Streaming tuning
├── tests/
│   ├── test_dynamic_graph.py            # 45 testes
│   ├── test_streaming.py                # 22 testes
│   └── test_model_adaptations.py        # 12 testes
└── configs/
    ├── NF-UNSW-NB15-v3.yaml             # Dataset configs
    └── ...
```
