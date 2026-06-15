# GraphIDS com Suporte a Grafos Dinâmicos — Guia Didático

## 1. Visão Geral do Problema

O GraphIDS original construía um **grafo estático** único contendo **todos os IPs e fluxos**
do dataset de uma só vez. Isso funciona para análise offline, mas é inviável para
redes reais onde:

- Novos dispositivos (IPs) aparecem constantemente
- Dispositivos saem da rede após um período
- Fluxos de rede têm tempo de vida finito (expiração)
- A detecção precisa acontecer em **tempo real**, não em lotes

### Exemplo Concreto

```
Rede doméstica:
  08:00 - Notebook A conecta (IP novo) → 3 fluxos DNS
  08:05 - Smartphone B conecta (IP novo) → 2 fluxos HTTP
  08:10 - Notebook A desconecta → arestas do A devem expirar
  08:15 - IoT C conecta → novo nó, zero histórico
```

O modelo precisa **detectar anomalias em cada fluxo** enquanto a topologia evolui.

---

## 2. Arquitetura da Solução

```
[Fluxos de Rede] ──► [DynamicGraph] ──► [GNN Encoder] ──► [Transformer AE] ──► [Score]
                           │                                                         │
                           ▼                                                         ▼
                    [Expiração]                                              [Threshold]
                    [Remoção]                                                [Adaptativo]
```

### 2.1 DynamicGraph (`utils/dynamic_graph.py`)

**O que é:** Estrutura de dados que mantém o grafo em evolução.

**Responsabilidades:**
- **Mapeamento dinâmico IP → node_id**: Quando um IP nunca antes visto aparece,
  um novo node_id é automaticamente alocado (incremento monotônico)
- **Armazenagem temporal**: Cada aresta guarda seu timestamp de criação
- **Expiração**: Arestas mais velhas que `max_edge_age_ms` são removidas
- **Remoção de nós**: Nós sem arestas por muito tempo são removidos
- **Snapshots**: Produz objetos `Data` do PyTorch Geometric para consumo pelo modelo

**Decisões arquiteturais:**

| Decisão | Motivo |
|---|---|
| Listas paralelas (`_edge_src`, `_edge_dst`, etc.) | Inserção O(1), acesso indexado rápido |
| `_edge_active: List[bool]` (soft delete) | Remoção O(1) sem reorganizar arrays |
| `adj_out / adj_in: Dict[int, List]` | Consulta de vizinhança O(grau) sem varredura |
| `node_map: Dict[str, int]` | Lookup de IP O(1) |
| Snapshot com remapeamento de IDs | Evita "buraco" de nós removidos no tensor final |

**Exemplo de uso:**
```python
dg = DynamicGraph(edge_feature_dim=10, max_edge_age_ms=3600000)
dg.add_edge("192.168.1.1", "10.0.0.1", [features...], timestamp=1000)
dg.add_edge("192.168.1.2", "10.0.0.1", [features...], timestamp=2000)
snapshot = dg.get_current_graph()  # → Data(x, edge_index, edge_attr, ...)
dg.remove_stale_edges(current_time=5000)  # remove arestas com > 1h
dg.remove_inactive_nodes(max_age_ms=7200000, current_time=5000)  # remove nós inativos
```

### 2.2 StreamingNetFlowDataset (`utils/dataloaders.py`)

**O que é:** Dataset que lê fluxos ordenados por tempo e produz uma sequência
de snapshots (janelas temporais).

**Como funciona:**
1. Lê o CSV completo (como o `NetFlowDataset` original)
2. Ordena por `FLOW_START_MILLISECONDS`
3. Divide o tempo em janelas de `window_size_ms` com stride de `stride_ms`
4. Para cada janela, adiciona os fluxos ao `DynamicGraph` e tira um snapshot
5. Armazena `self.windows: List[Data]` — uma lista de grafos, um por janela

**Diferença do `NetFlowDataset` original:**

| Aspecto | NetFlowDataset | StreamingNetFlowDataset |
|---|---|---|
| Saída | 3 grafos (train/val/test) | N grafos (janelas temporais) |
| Mapeamento de nós | Global (todos os splits) | Dinâmico (por janela) |
| Ordem temporal | Opcional | Obrigatória |
| Uso principal | Treino offline batch | Treino temporal + streaming |

### 2.3 SAGELayer Aprimorado (`models/graphids.py`)

**O que mudou:**

**Pesos temporais** — parâmetro `temporal_weights: Optional[Tensor]`:
```python
def message(self, edge_attr, edge_temporal_weights=None):
    if edge_temporal_weights is not None:
        return edge_attr * edge_temporal_weights.unsqueeze(-1)
    return edge_attr
```
Isso permite que arestas mais antigas contribuam menos para o embedding do nó.
O peso pode ser `exp(-λ * idade)` — quanto mais velha a aresta, menor o impacto.

**Propagação seletiva** — parâmetro `node_mask: Optional[Tensor]`:
```python
if node_mask is not None:
    edge_mask = máscara de arestas incidentes aos nós em node_mask
    edge_index = edge_index[:, edge_mask]
    edge_attr = edge_attr[edge_mask]
```
Permite propagar apenas nos nós afetados por mudanças recentes, evitando
recalcular embeddings de nós não modificados.

**ColdStartNodeInitializer**:
```python
class ColdStartNodeInitializer:
    def initialize_nodes(self, node_ids, node_embeddings, edge_index, num_nodes):
        # Preenche NaN de nós novos com média dos vizinhos
        # Se sem vizinhos, usa embedding default aprendido
```

### 2.4 StreamingDetector (`inference/streaming.py`)

**Detecção em tempo real, fluxo a fluxo:**

```
process_flow(src_ip, dst_ip, features, timestamp):
    1. add_edge() ao DynamicGraph
    2. get_current_graph() → Data
    3. model.encoder(data) → embedding da nova aresta
    4. buffer.add(embedding) → janela deslizante
    5. model.transformer(window) → reconstrução
    6. erro de reconstrução → anomaly_score
    7. thresholder.add_score(score) → threshold adaptativo
    8. score > threshold → ALERTA
```

**SlidingWindowBuffer**: Buffer circular thread-safe que mantém as últimas N
arestas para o Transformer AE processar.

**AdaptiveThreshold**: Calcula threshold com MAD (Median + multiplier × MAD)
sobre uma janela deslizante dos scores recentes. Opcionalmente usa validation_f1
se houver labels disponíveis.

**Processamento em lote (`process_batch`):** Otimizado para múltiplos fluxos:
- Único forward pass do GNN para todas as arestas novas
- Único forward pass do Transformer AE para todas as janelas empilhadas
- Reduz latência significativamente vs. processar fluxo a fluxo

### 2.5 Treinamento Online (`utils/trainers.py`)

**train_online()** — Fine-tuning com experience replay:
```python
def train_online(model, new_data, replay_buffer, ...):
    # 1. Amostra batch do novo_data
    # 2. Amostra batch aleatório do replay_buffer
    # 3. Loss combinada = loss_novo + loss_replay
    # 4. Backpropagation
    # 5. Repete por num_steps
```
Isso evita **catastrophic forgetting** — o modelo não esquece ataques antigos
ao aprender novos padrões.

**train_temporal_windows()** — Treino sequencial por janelas:
```python
def train_temporal_windows(model, dataset, ...):
    for window_idx in range(num_windows):
        # Treina na janela atual
        # Valida na próxima janela (se existir)
        # Adiciona janela ao replay_buffer
        # Early stopping por janela
```

---

## 3. Fluxo de Treinamento (Modo Streaming)

O pipeline completo ao executar `--streaming`:

```
1. StreamingNetFlowDataset carrega dados em N janelas temporais
2. Treina modelo nas primeiras 60% das janelas (treino inicial)
3. Para cada janela restante (40%):
   a. Valida modelo na janela (inferência)
   b. Calcula F1 e PR-AUC
   c. Atualiza threshold com EMA (alpha=0.1)
   d. Loga métricas no wandb
4. Retorna modelo treinado
```

### Parâmetros-chave

| Parâmetro | Default | Descrição |
|---|---|---|
| `window_size_ms` | 3600000 (1h) | Tamanho da janela temporal |
| `stride_ms` | 1800000 (30min) | Passo entre janelas (sobreposição) |
| `max_edge_age_ms` | 3600000 (1h) | TTL máximo de uma aresta |
| `max_nodes` | 100000 | Limite de segurança de nós |
| `threshold_method` | "mad" | MAD ou validation_f1 |
| `adaptation_window` | 1000 | Janela de scores para threshold |

---

## 4. Dataset Utilizado

O projeto trabalha com datasets de **NetFlow** no formato CSV, especificamente
as séries **NF-UNSW-NB15**, **NF-CSE-CIC-IDS2018**, e **NF-ToN-IoT**
(nas variantes v2 e v3).

Cada linha do CSV representa um **fluxo de rede** com:
- `IPV4_SRC_ADDR`, `IPV4_DST_ADDR` → nós do grafo
- Dezenas de features estatísticas (bytes, pacotes, portas, etc.) → `edge_attr`
- `FLOW_START_MILLISECONDS` → timestamp para ordenação temporal
- `Label` (0/1) → benigno/maligno

**Para testar sem dataset real:** O `conftest.py` contém fixtures que geram
datasets sintéticos (`ToyNF`) com as mesmas colunas.

---

## 5. Exemplo de Uso Completo

```bash
# Treino e teste no modo streaming
python main.py \
  --dataset NF-UNSW-NB15-v3 \
  --data_dir ./data \
  --streaming \
  --window_size 512 \
  --edim_out 8 \
  --ae_embedding_dim 16 \
  --num_epochs 50 \
  --learning_rate 1e-3

# Modo batch tradicional (inalterado)
python main.py \
  --dataset NF-UNSW-NB15-v3 \
  --data_dir ./data \
  --split_mode temporal
```

---

## 6. Implicações de Desempenho

### Eficiência

| Operação | Complexidade | Frequência |
|---|---|---|
| add_edge() | O(1) amortizado | Cada fluxo |
| remove_stale_edges() | O(E) | Periódica |
| Snapshot | O(E + N) | A cada janela |
| Encoder forward | O(E + N) | A cada batch |
| Transformer AE | O(W²) | A cada janela |

### Gargalo Identificado

O **Transformer Autoencoder** tem complexidade O(W²) devido à self-attention,
onde W = window_size (default 512). Para processamento em tempo real, este é
o principal gargalo — cada execução envolve 512² ≈ 262K operações de atenção.

Para mitigação:
- Usar `window_size` menor (ex: 128) para latência menor
- Usar `step_percent < 1.0` para sobreposição de janelas
- Processamento em lote (`process_batch`) reduz overhead

### Segurança

- **Cold start**: Novos nós começam com embedding zero/inicializado → primeiros
  fluxos podem ter detecção menos precisa
- **Graph poisoning**: Limitamos `max_nodes` e each nó tem grau máximo implícito
- **Threshold drift**: O threshold adaptativo pode ser lentamente deslocado por
  ataques. Usamos EMA (alpha=0.1) para suavizar mudanças.
- **Flapping**: Oscilação de nós entrando/saindo é mitigada pelo TTL e remoção
  apenas após inatividade prolongada.

---

## 7. Testes

```bash
# Testes de regressão (config + preprocessing)
uv run python -m pytest tests/test_config.py tests/test_preprocessing.py -v

# Testes de componentes dinâmicos (verificação manual)
uv run python -c "
from models.graphids import SAGELayer, GraphIDS, ColdStartNodeInitializer
from utils.dynamic_graph import DynamicGraph
from inference.streaming import SlidingWindowBuffer, AdaptiveThreshold
from utils.trainers import update_threshold_online
# ... testes de integração ...
"

# Teste completo (requer pyg-lib com GLIBC >= 2.33)
uv run python -m pytest tests/ -v
```

---

## 8. Resumo das Decisões Arquiteturais

1. **DynamicGraph como estrutura central** — toda a gestão de topologia
   (add/remove/expire) concentrada em uma classe, separada do modelo e do dataset.
2. **Soft delete com `_edge_active`** — remoção O(1) sem fragmentação de memória.
3. **Snapshots com remapeamento** — nós removidos não criam buracos nos tensores.
4. **Pesos temporais no MessagePassing** — integração natural com o PyG,
   sem modificar a arquitetura do modelo.
5. **Thread safety** — `StreamingDetector`, `SlidingWindowBuffer`,
   e `AdaptiveThreshold` usam locks para suportar inferência concorrente.
6. **Backward compatibility** — todos os parâmetros novos têm defaults None;
   código existente continua funcionando sem alterações.
7. **Experience replay** no treinamento online — evita catastrophic forgetting
   sem precisar de regularização complexa (EWC).
8. **Threshold adaptativo com EMA** — mudanças graduais evitam oscilações
   bruscas na taxa de alertas.
