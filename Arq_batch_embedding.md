# Arquitectura: Extractive Question Answering

## Diagrama General

```
┌──────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐     ┌────────────┐
│          │     │               │     │              │     │              │     │            │
│ PREGUNTA │────▶│   RETRIEVER   │────▶│   PINECONE   │────▶│   READER     │────▶│ RESPUESTA  │
│          │     │               │     │              │     │              │     │            │
└──────────┘     └───────────────┘     └──────────────┘     └──────────────┘     └────────────┘
                      │                      │                     │
                      │                      │                     │
                 "multi-qa-            base de datos         "deepset/
                 MiniLM-L6-            vectorial              electra-base-
                 cos-v1"               (cosine)               squad2"
```

---

## Flujo Paso a Paso

```
                           FASE 1: INDEXACIÓN (se hace UNA sola vez)
                           ──────────────────────────────────────────
                           
   ┌──────────┐      ┌──────────────┐      ┌──────────────┐
   │ contexto │─────▶│  RETRIEVER   │─────▶│  PINECONE    │
   │ (texto)  │      │  .encode()   │      │  .upsert()   │
   └──────────┘      └──────────────┘      └──────────────┘
                           │                      │
                           ▼                      ▼
                     [0.12, -0.34,          guarda: id
                      0.87, ...]            + vector (384 nums)
                       ↑                    + metadata (title, texto)
                      384 números


                           FASE 2: CONSULTA (cada vez que hacés una pregunta)
                           ──────────────────────────────────────────────────

┌──────────┐      ┌──────────────┐      ┌──────────────┐      ┌──────────────┐      ┌────────────┐
│          │      │              │      │              │      │              │      │            │
│ "How     │─────▶│  RETRIEVER   │─────▶│  PINECONE    │─────▶│   READER     │─────▶│ "691,000   │
│  much    │      │  .encode()   │      │  .query()    │      │  pipeline()  │      │  bbl/d"    │
│  oil...?"│      │              │      │              │      │              │      │            │
└──────────┘      └──────────────┘      └──────────────┘      └──────────────┘      └────────────┘
                        │                      │                      │
                        ▼                      ▼                      ▼
                  convierto la            busco los top_k          por cada contexto,
                  pregunta en             vectores más             extraigo la respuesta
                  384 números             parecidos (cosine)       y me quedo con la
                                          ─────────────────       de mayor score
                                          devuelve: id +
                                          vector + metadata

```

---

## El Loop de Batching (detalle de la Fase 1)

```
df (dataframe con 1000+ contextos)
│
├── Lote 0: filas 0..63  ──▶ encode(64 textos) ──▶ 64 vectores ──▶ upsert a Pinecone
├── Lote 1: filas 64..127 ──▶ encode(64 textos) ──▶ 64 vectores ──▶ upsert a Pinecone
├── Lote 2: filas 128..191──▶ encode(64 textos) ──▶ 64 vectores ──▶ upsert a Pinecone
│   ...
└── Último lote ────────────▶ encode(lo que sobre) ──▶ upsert a Pinecone

¿Por qué en lotes?
  - Mandar 1 por 1 → 1000 llamadas a Pinecone = lentísimo
  - Mandar todo junto → posible timeout o memory error
  - Batch de 64 → equilibrio entre velocidad y estabilidad
```

---

## ¿Qué hace cada modelo?

| Componente | Modelo | Entrada | Salida | Rol |
|------------|--------|---------|--------|-----|
| **Retriever** | `multi-qa-MiniLM-L6-cos-v1` | Texto (pregunta o contexto) | Vector de 384 números | Búsqueda semántica: encontrar contextos relevantes |
| **Pinecone** | - | Vector de 384 números | top_k vectores + metadata | Almacenar y recuperar rápido (sin esto habría que comparar contra todos cada vez) |
| **Reader** | `deepset/electra-base-squad2` | Pregunta + 1 contexto | `{answer, score, start, end}` | Extracción fina: pinpoint la respuesta dentro del texto |

---

## Analogía para no olvidarlo

```
Retriever = buscador de Google
    ↳ Le das "receta pizza" y te devuelve 5 links relevantes

Pinecone = el índice de Google
    ↳ La base de datos gigante donde Google guardó todas las páginas indexadas

Reader = tus ojos leyendo la receta
    ↳ De los 5 links, abrís uno y encontrás "200g de harina"
```

---

## El código equivalente

```python
# FASE 1: Indexar (se hace una vez)
for i in range(0, len(df), 64):
    batch = df.iloc[i : i+64]                           # corto 64 filas
    emb   = retriever.encode(batch["context"].tolist())  # textos → vectores
    meta  = [{"title": r.title, "context": r.context}
             for r in batch.itertuples()]                # guardo de qué texto vino
    ids   = [str(j) for j in range(i, i+64)]             # IDs únicos
    index.upsert(vectors=zip(ids, emb, meta))            # subo a Pinecone


# FASE 2: Preguntar (cada vez)
def get_context(question, top_k):
    xq = retriever.encode(question)           # pregunta → vector
    xc = index.query(vector=xq, top_k=top_k)  # busco en Pinecone
    return [match["metadata"]["context"]       # extraigo solo el texto
            for match in xc["matches"]]


def extract_answer(question, contexts):
    results = [reader(question=q, context=c) for c in contexts]  # reader extrae
    return sorted(results, key=lambda r: r["score"], reverse=True)  # mejor primero
```
