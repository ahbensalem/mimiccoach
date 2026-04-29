# MimicCoach — Self-Coaching by Pose-Embedding Lookup

A user records a phone video of themselves performing a tennis serve, skateboard trick, golf swing, or yoga pose. The app extracts a pose-sequence embedding (MediaPipe/MoveNet → temporal aggregation), queries Qdrant against a curated library of professional reference clips, and overlays a side-by-side diff with the closest pro match — filtered by user-chosen skill level, body type, or sport. The novelty is using Qdrant's late-interaction/multivector storage for time-sliced pose tokens (similar to how ColBERT/ColPali use it for documents) so that retrieval is sensitive to which phase of the motion is off, not just an averaged whole-sequence embedding. No public Qdrant demo currently does pose-based motion retrieval.

## Architecture

```mermaid
flowchart TB
    subgraph User["User Capture"]
        U1[Record phone video<br/>tennis serve / golf swing / yoga]
        U2[Select sport + skill level + body type]
    end

    subgraph PoseExtract["Pose Extraction"]
        PE1[MediaPipe / MoveNet]
        PE2[Per-frame keypoints]
        PE3[Temporal segmentation<br/>into motion phases]
        U1 --> PE1 --> PE2 --> PE3
    end

    subgraph TokenEncode["Late-Interaction Token Embedding"]
        T1[Phase 1: setup tokens]
        T2[Phase 2: backswing tokens]
        T3[Phase 3: contact tokens]
        T4[Phase 4: follow-through tokens]
        PE3 --> T1
        PE3 --> T2
        PE3 --> T3
        PE3 --> T4
    end

    subgraph ProLib["Pro Reference Library"]
        L1[Curated pro clips]
        L2[Pose-token sequences<br/>ColBERT-style]
        L3[Payload: sport, level,<br/>body type, athlete]
        L1 --> L2
    end

    subgraph QdrantStore["Qdrant Multivector Storage"]
        Q[(Reference Points<br/>multivector: token-per-phase<br/>+ MaxSim late interaction)]
        L2 --> Q
        L3 --> Q
    end

    subgraph Retrieval["Phase-Aware Retrieval"]
        R1[Query tokens from user video]
        R2[Filter: skill level,<br/>body type, sport]
        R3[Multivector search<br/>MaxSim across phases]
        T1 --> R1
        T2 --> R1
        T3 --> R1
        T4 --> R1
        U2 --> R2
        R1 --> R3
        R2 --> R3
    end

    R3 -->|search| Q
    Q -->|closest pro match<br/>+ per-phase scores| Diff

    subgraph DiffEngine["Diff & Coaching"]
        Diff[Per-phase similarity scores]
        Diff --> D1[Identify weakest phase]
        Diff --> D2[Side-by-side overlay]
        Diff --> D3[Skeleton diff visualization]
        D1 --> Out[Targeted coaching tips]
        D2 --> Out
        D3 --> Out
    end

    style Q fill:#dc2626,color:#fff
    style R3 fill:#dc2626,color:#fff
    style D1 fill:#fbbf24
```

## Qdrant Features Showcased

- **Late-interaction multivector storage** — ColBERT/ColPali-style token-per-phase embeddings instead of a single averaged vector
- **MaxSim scoring** — per-phase similarity makes retrieval sensitive to *which* part of the motion diverges
- **Payload filtering** — skill level, body type, and sport narrow the reference pool before similarity ranking
