# Stadium Pulse — Real-Time Search of Crowd & Cultural Soundscapes

A streaming pipeline that ingests live audio from sports broadcasts, opera houses, religious processions, political rallies, and concerts, segments it into ~5-second windows, encodes each with a CLAP or PaSST audio model, and upserts vectors into Qdrant in real time. A user can submit a query clip ("a 90,000-person collective gasp," "a slow-build standing ovation," "a single voice silencing a stadium," "the second between a free-kick whistle and a goal") and get cross-event matches. Filters by venue, sport/genre, decibel range, and date make it both a sports analytics tool (which atmospheres precede comebacks?) and a cultural-anthropology tool. This idea uniquely showcases Qdrant's instant indexing ("vectors are searchable the moment they're added"), quantization for low-RAM long-tail audio storage, and the rare combination of streaming ingestion + ANN search that very few open demos exhibit.

## Architecture

```mermaid
flowchart TB
    subgraph Sources["Live Audio Sources"]
        S1[Sports broadcasts]
        S2[Opera houses]
        S3[Religious processions]
        S4[Political rallies]
        S5[Concerts]
    end

    subgraph Stream["Streaming Pipeline"]
        ST1[Audio capture<br/>RTSP / HLS / mic feeds]
        ST2[5-second sliding windows]
        ST3[VAD + decibel measurement]
        S1 --> ST1
        S2 --> ST1
        S3 --> ST1
        S4 --> ST1
        S5 --> ST1
        ST1 --> ST2 --> ST3
    end

    subgraph Encode["Audio Embedding"]
        E1[CLAP / PaSST model]
        E2[512-dim audio vector]
        ST3 --> E1 --> E2
    end

    subgraph Meta["Payload Enrichment"]
        M1[Venue + GPS]
        M2[Sport / genre / event type]
        M3[Decibel range]
        M4[Timestamp + date]
        M5[Game state<br/>score, minute, period]
    end

    subgraph QdrantRT["Qdrant — Real-Time Collection"]
        Q[(Audio Window Points<br/>instant indexing<br/>+ scalar quantization)]
        E2 -->|upsert| Q
        M1 --> Q
        M2 --> Q
        M3 --> Q
        M4 --> Q
        M5 --> Q
    end

    subgraph QueryUI["Query Interface"]
        QU1[Upload query clip<br/>'collective gasp']
        QU2[Text-to-audio prompt<br/>via CLAP joint embedding]
        QU3[Filter: venue, genre,<br/>dB range, date]
        QU1 --> QE
        QU2 --> QE
        QE[Encode query → vector]
        QE --> QS[ANN search<br/>+ payload filter]
        QU3 --> QS
    end

    QS -->|search| Q
    Q -->|matched windows| Results

    subgraph Apps["Downstream Applications"]
        Results --> A1[Sports analytics<br/>'atmospheres before comebacks']
        Results --> A2[Cultural anthropology<br/>cross-event patterns]
        Results --> A3[Highlight reels<br/>auto-clip from sound]
        Results --> A4[Live alerts<br/>detect rare crowd moments]
    end

    style Q fill:#dc2626,color:#fff
    style QS fill:#dc2626,color:#fff
    style ST2 fill:#fbbf24
```

## Qdrant Features Showcased

- **Instant indexing** — vectors are searchable the moment they are upserted from the live stream, no batch reindex
- **Scalar / product quantization** — keeps RAM low across a long-tail archive of millions of audio windows
- **Streaming ingestion + ANN search** — sustained upsert + query throughput on the same collection, a rare combination in public demos
- **Payload-indexed filtering** — venue, genre, decibel range, date, and game state queried alongside the vector
- **Multimodal CLAP queries** — joint text-and-audio embedding lets users search by natural-language prompt or by example clip
