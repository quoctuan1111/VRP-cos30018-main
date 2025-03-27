sequenceDiagram
    participant MRA
    participant DA
    MRA->>DA: CAPACITY_REQUEST
    Note right of DA: Process Request
    DA-->>MRA: CAPACITY_RESPONSE