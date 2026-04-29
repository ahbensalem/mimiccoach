# TacticalTwin — Play Trajectory Retrieval for Coaches and Fans

Index basketball, football, or hockey possessions as multi-vector points: one vector for the offensive player trajectory, one for the defensive trajectory, one for the ball, plus a payload of game state (score differential, time remaining, opponent, zone of court/pitch). A coach draws a play on a tactics board (or selects an existing clip), and Qdrant returns the most similar real possessions filtered by "down by ≤2 in the last 5 minutes" or "vs. zone defense." This operationalizes the play2vec / SportVU research that has lived only in Sloan Conference papers and turns it into an interactive demo. Showcases multivector queries, payload-indexed filtering, and the Recommend API (positive/negative examples = "make it more like this attacking style, less like that one").

## Architecture

```mermaid
flowchart TB
    subgraph Input["Coach / Fan Input"]
        A1[Draw play on tactics board]
        A2[Select existing clip]
        A3[Filters: score diff, time, defense type]
    end

    subgraph Ingest["Ingestion Pipeline"]
        B1[SportVU / tracking data]
        B2[Possession segmentation]
        B3[Trajectory extraction<br/>play2vec encoder]
        B1 --> B2 --> B3
    end

    subgraph Vectors["Multi-Vector Encoding"]
        V1[Offensive trajectory vector]
        V2[Defensive trajectory vector]
        V3[Ball trajectory vector]
        B3 --> V1
        B3 --> V2
        B3 --> V3
    end

    subgraph Payload["Payload Metadata"]
        P1[Score differential]
        P2[Time remaining]
        P3[Opponent]
        P4[Zone of court/pitch]
    end

    subgraph Qdrant["Qdrant Collection"]
        Q[(Possession Points<br/>3 named vectors + payload)]
        V1 --> Q
        V2 --> Q
        V3 --> Q
        P1 --> Q
        P2 --> Q
        P3 --> Q
        P4 --> Q
    end

    subgraph Query["Query Engine"]
        QE1[Encode drawn play]
        QE2[Multi-vector search]
        QE3[Payload-indexed filter<br/>'down by ≤2, last 5 min']
        QE4[Recommend API<br/>+ like attacking style<br/>− unlike this style]
        A1 --> QE1
        A2 --> QE1
        A3 --> QE3
        QE1 --> QE2
        QE2 --> QE3
        QE3 --> QE4
    end

    QE4 -->|search| Q
    Q -->|top-k possessions| R[Ranked Similar Plays]

    subgraph Output["Coach Dashboard"]
        R --> O1[Side-by-side video clips]
        R --> O2[Trajectory overlay]
        R --> O3[Outcome statistics]
    end

    style Q fill:#dc2626,color:#fff
    style QE4 fill:#dc2626,color:#fff
```

## Qdrant Features Showcased

- **Named multi-vector points** — separate vectors for offense, defense, and ball trajectories on a single point
- **Payload-indexed filtering** — game-state filters (score diff, time, opponent, zone) executed alongside vector search
- **Recommend API** — positive/negative examples to steer search toward an attacking style or away from another
