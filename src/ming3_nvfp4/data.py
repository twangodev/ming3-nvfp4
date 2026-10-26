from __future__ import annotations

import random
from pathlib import Path

ASR_PROMPT = ("Please recognize the language of this speech and transcribe it, "
              "placing a `\t` between the identified language and the transcribe.")


def fleurs_examples(path: Path, seed: int):
    """Bounded shuffle over local Parquet, with no HTTP readers alive at shutdown."""
    import pyarrow.parquet as pq

    rng = random.Random(seed)
    buffer = []
    with pq.ParquetFile(path) as parquet:
        for batch in parquet.iter_batches(batch_size=32, use_threads=False):
            for row in batch.to_pylist():
                if len(buffer) < 128:
                    buffer.append(row)
                    continue
                index = rng.randrange(len(buffer))
                yield buffer[index]
                buffer[index] = row
        rng.shuffle(buffer)
        yield from buffer
